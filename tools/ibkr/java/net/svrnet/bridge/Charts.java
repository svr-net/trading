package net.svrnet.bridge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The page's chart of an order (GET /chart) and the Lab's bars (GET /bars.csv), as bridge.py: IBKR's daily bars from the file the bars
 * fetch keeps (ibkr-<MARKET>.csv; the store's when there is none yet), the store's own bars for a share IBKR's file lacks or before IBKR's
 * first day, and the share's profile. Prices go only to the page that holds the bridge's key, never to the published page (the data licence).
 */
final class Charts {
    static final Map<String, List<String>> PROFILE_EXCHANGES = Map.of("UK", List.of("LSE"), "US", List.of("NYSE", "NASDAQ"),
            "EU", List.of("PAR", "AMS", "BRU", "LIS", "MSE", "FRA"), "APAC", List.of("ASX", "SGX"));
    static final Map<String, Object[]> CACHE = new ConcurrentHashMap<>();  // market -> {mtime, {ticker: bars}}
    static final Map<String, Object[]> STORE_BARS = new ConcurrentHashMap<>();  // exchange/code -> {when (ms), bars}
    static final Map<String, Object[]> PROFILES = new ConcurrentHashMap<>();  // exchange -> {when (ms), {code: profile}}
    static final long HALF_DAY = 12L * 3600 * 1000;

    static Path kept(String market) {
        return Path.of(Auto.HISTORY, "ibkr-" + market + ".csv");
    }

    /** The kept file, from the store when there is none here yet; whether there is one now. */
    static boolean ensureKept(String market) throws IOException {
        Path p = kept(market);
        if (Files.exists(p) && Files.size(p) > 100) return true;
        byte[] k = Store.getDaily(market);
        if (k == null || k.length < 100) return false;
        Files.createDirectories(p.getParent());
        Files.write(p, k);
        return true;
    }

    static Map<String, Object> bar(String d, double o, double h, double l, double c, double v) {
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("d", d);
        b.put("o", o);
        b.put("h", h);
        b.put("l", l);
        b.put("c", c);
        b.put("v", v);
        return b;
    }

    static double d(Object o) {
        return ((Number) o).doubleValue();
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> bars(String market, String ticker, int days) throws IOException {
        int want = Math.max(1, Math.min(days, 520));
        if (!ensureKept(market)) return tail(storeBars(market, ticker), want);
        Path p = kept(market);
        long mtime = Files.getLastModifiedTime(p).toMillis();
        Object[] hit = CACHE.get(market);
        if (hit == null || (long) hit[0] != mtime) {
            Map<String, List<Map<String, Object>>> by = new LinkedHashMap<>();
            for (String[] r : Csv.read(p)) {
                if (r.length < 7 || r[0].equals("date")) continue;
                try {
                    by.computeIfAbsent(r[1], k -> new ArrayList<>()).add(bar(Auto.day(r[0]), Double.parseDouble(r[2]), Double.parseDouble(r[3]), Double.parseDouble(r[4]),
                            Double.parseDouble(r[5]), r[6].isEmpty() ? 0 : Double.parseDouble(r[6])));
                } catch (NumberFormatException e) {
                    // a bad row: left out
                }
            }
            for (List<Map<String, Object>> b : by.values()) b.sort((a, c) -> ((String) a.get("d")).compareTo((String) c.get("d")));
            hit = new Object[] {mtime, by};
            CACHE.put(market, hit);
        }
        Map<String, List<Map<String, Object>>> by = (Map<String, List<Map<String, Object>>>) hit[1];
        String code = ticker.substring(ticker.lastIndexOf(':') + 1);
        List<Map<String, Object>> b = by.get(ticker);
        if (b == null || b.isEmpty()) b = by.get(code);
        if (b == null || b.isEmpty()) {
            b = List.of();
            for (Map.Entry<String, List<Map<String, Object>>> e : by.entrySet()) {
                if (e.getKey().substring(e.getKey().lastIndexOf(':') + 1).equals(code)) {
                    b = e.getValue();
                    break;
                }
            }
        }
        if (b.isEmpty()) b = storeBars(market, ticker);
        else if (b.size() < want) b = withOlder(b, storeBars(market, ticker));
        return tail(b, want);
    }

    static List<Map<String, Object>> tail(List<Map<String, Object>> b, int n) {
        return new ArrayList<>(b.subList(Math.max(0, b.size() - n), b.size()));
    }

    /** IBKR's bars with the store's from before IBKR's first day, scaled by IBKR's close over the store's on the first day both have. */
    static List<Map<String, Object>> withOlder(List<Map<String, Object>> bars, List<Map<String, Object>> older) {
        if (bars.isEmpty() || older.isEmpty()) return bars;
        String first = (String) bars.get(0).get("d");
        Map<String, Double> sc = new LinkedHashMap<>();
        for (Map<String, Object> b : older) sc.put((String) b.get("d"), d(b.get("c")));
        Map<String, Object> same = null;
        for (Map<String, Object> b : bars) if (sc.containsKey(b.get("d"))) {
            same = b;
            break;
        }
        double k = same != null && sc.get(same.get("d")) > 0 ? d(same.get("c")) / sc.get(same.get("d")) : 1.0;
        if (!(0.004 < k && k < 250)) return bars;
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> b : older) {
            if (((String) b.get("d")).compareTo(first) < 0)
                out.add(bar((String) b.get("d"), d(b.get("o")) * k, d(b.get("h")) * k, d(b.get("l")) * k, d(b.get("c")) * k, d(b.get("v"))));
        }
        out.addAll(bars);
        return out;
    }

    static List<String> exchanges(String market, String ticker) {
        return ticker.contains(":") ? List.of(ticker.substring(0, ticker.indexOf(':'))) : PROFILE_EXCHANGES.getOrDefault(market, List.of());
    }

    /** A share's daily bars from the store (store/<EXCHANGE>/<CODE>.json; an index ETF's store/series/<EXCHANGE>_<CODE>.json), kept 12 hours. */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> storeBars(String market, String ticker) {
        String code = ticker.substring(ticker.lastIndexOf(':') + 1);
        for (String e : exchanges(market, ticker)) {
            String key = e + "/" + code;
            Object[] hit = STORE_BARS.get(key);
            if (hit == null || System.currentTimeMillis() - (long) hit[0] > HALF_DAY) {
                List<Map<String, Object>> bars = new ArrayList<>();
                try {
                    byte[] body = Store.get("store/" + e + "/" + code + ".json");
                    if (body == null) body = Store.get("store/series/" + e + "_" + code + ".json");
                    Map<String, Object> all = body == null ? Map.of() : JsonParse.object(new String(body, StandardCharsets.UTF_8));
                    for (Map.Entry<String, Object> x : new TreeMap<>(all).entrySet()) {
                        if (!(x.getValue() instanceof Map)) continue;
                        Map<String, Object> b = (Map<String, Object>) x.getValue();
                        boolean ok = true;
                        for (String k : List.of("o", "h", "l", "c")) ok &= b.get(k) instanceof Number && d(b.get(k)) > 0;
                        if (ok) bars.add(bar(Auto.day(x.getKey()), d(b.get("o")), d(b.get("h")), d(b.get("l")), d(b.get("c")), b.get("v") instanceof Number ? d(b.get("v")) : 0));
                    }
                } catch (IOException | RuntimeException ex) {
                    bars = new ArrayList<>();
                }
                hit = new Object[] {System.currentTimeMillis(), bars};
                STORE_BARS.put(key, hit);
            }
            List<Map<String, Object>> b = (List<Map<String, Object>>) hit[1];
            if (!b.isEmpty()) return b;
        }
        return List.of();
    }

    /** The share's name, sector, industry, description, market value, P/E and dividend yield (the store's profiles/<EXCHANGE>.json); {} when none. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> profile(String market, String ticker) {
        String code = ticker.substring(ticker.lastIndexOf(':') + 1);
        for (String e : exchanges(market, ticker)) {
            Object[] hit = PROFILES.get(e);
            if (hit == null || System.currentTimeMillis() - (long) hit[0] > HALF_DAY) {
                Map<String, Map<String, Object>> rows = new LinkedHashMap<>();
                try {
                    byte[] body = Store.get("profiles/" + e + ".json");
                    Object all = JsonParse.parse(body == null ? "[]" : new String(body, StandardCharsets.UTF_8));
                    if (all instanceof List) for (Object r0 : (List<Object>) all) {
                        if (!(r0 instanceof Map)) continue;
                        Map<String, Object> r = (Map<String, Object>) r0;
                        if (r.get("code") == null || "".equals(r.get("code"))) continue;
                        Map<String, Object> p = new LinkedHashMap<>();
                        for (String k : List.of("name", "sector", "industry", "description", "marketCap", "pe", "dividendYield")) {
                            Object v = r.get(k);
                            if (v != null && !"".equals(v)) p.put(k, v);
                        }
                        rows.put(String.valueOf(r.get("code")), p);
                    }
                } catch (IOException | RuntimeException ex) {
                    rows = new LinkedHashMap<>();
                }
                hit = new Object[] {System.currentTimeMillis(), rows};
                PROFILES.put(e, hit);
            }
            Map<String, Map<String, Object>> rows = (Map<String, Map<String, Object>>) hit[1];
            if (rows.containsKey(code)) {
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("exchange", e);
                out.putAll(rows.get(code));
                return out;
            }
        }
        return new LinkedHashMap<>();
    }

    private Charts() {
    }
}
