package net.svrnet.bridge;

import com.ib.client.Bar;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Shadow mode: every 5 minutes of each market's session, each of the day's orders in its state (not filled yet: the
 * price's distance above its limit; bought: its move since the buy; both in the share's daily volatilities, at the
 * last finished 5-minute bar) and the action the learnt intraday policy (config/intraday-policy.json, from
 * tools/intraday_policy.py) would take there. Only recorded (history/shadow/MARKET/DATE.jsonl, GET /shadow): no order
 * is changed. A market without a policy yet records its states alone.
 */
final class Shadow {
    static final String POLICY_URL = Bridge.env("POLICY_URL", "https://raw.githubusercontent.com/svr-net/trading/main/config/intraday-policy.json");
    static volatile Map<String, Object> policies = Map.of();
    static volatile String policyNote = "not read yet";
    static long policyAt;
    static final Map<String, String> SLOT = new ConcurrentHashMap<>();       // market -> the last 5-minute slot checked
    static final Map<String, List<Map<String, Object>>> DAY = new ConcurrentHashMap<>();  // market -> the day's rows
    static final Map<String, String> NOTE = new ConcurrentHashMap<>();       // market -> why the last check had nothing

    static void loop() {
        while (true) {
            if (policyAt == 0 || Auto.ago(policyAt) >= 3600) readPolicy();
            for (String m : Auto.AUTO) {
                try {
                    tick(m);
                } catch (RuntimeException e) {
                    NOTE.put(m, "failed: " + e.getMessage());
                    Auto.err("shadow " + m + ": " + e.getMessage());
                }
            }
            Auto.beat("shadow");
            Recordings.sleep(20);
        }
    }

    /** The policies, hourly (a failed read keeps the last). */
    @SuppressWarnings("unchecked")
    static void readPolicy() {
        policyAt = System.nanoTime();
        try {
            HttpURLConnection c = (HttpURLConnection) new URL(POLICY_URL).openConnection();
            c.setConnectTimeout(20_000);
            c.setReadTimeout(20_000);
            c.setRequestProperty("User-Agent", "ofm-bridge");
            c.setRequestProperty("Cache-Control", "no-cache");
            if (c.getResponseCode() != 200) throw new IOException("HTTP " + c.getResponseCode());
            try (InputStream in = c.getInputStream()) {
                Map<String, Object> j = JsonParse.object(new String(in.readAllBytes(), StandardCharsets.UTF_8));
                policies = j.get("markets") instanceof Map ? (Map<String, Object>) j.get("markets") : Map.of();
            }
            policyNote = "read " + ZonedDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm z")) + ": " + String.join(", ", policies.keySet());
        } catch (IOException | RuntimeException e) {
            policyNote = "not read (" + e.getMessage() + "); the last kept";
        }
    }

    /** Once in each 5-minute slot of the session, a minute into it (its bar finished), from 6 minutes after the open to the close step. */
    static void tick(String m) {
        ZonedDateTime now = Auto.now(m);
        if (now.getDayOfWeek().getValue() >= 6) return;
        Object[] o = Holds.OPEN.get(m);
        ZonedDateTime open = now.withHour((int) o[1]).withMinute((int) o[2]).withSecond(0).withNano(0);
        ZonedDateTime close = Auto.at(now, Auto.TIMES.get(m)[1], 0);
        if (now.isBefore(open.plusMinutes(6)) || !now.isBefore(close)) return;
        long mins = Duration.between(open, now).toMinutes();
        if (mins % 5 < 1) return;
        String slot = now.toLocalDate() + " " + (mins / 5);
        if (slot.equals(SLOT.get(m))) return;
        SLOT.put(m, slot);
        check(m, now);
    }

    @SuppressWarnings("unchecked")
    static void check(String m, ZonedDateTime now) {
        String today = now.toLocalDate().toString();
        Map<String, Object> book;
        try {
            book = Auto.orders(m);
        } catch (IOException | RuntimeException e) {
            NOTE.put(m, "no orders file: " + e.getMessage());
            return;
        }
        if (!(book.get("days") instanceof Number) || ((Number) book.get("days")).intValue() != 1) {
            NOTE.put(m, "not a same-day market");
            return;
        }
        List<Map<String, Object>> orders = new ArrayList<>();
        if (book.get("orders") instanceof List) for (Object x : (List<Object>) book.get("orders")) if (x instanceof Map) orders.add((Map<String, Object>) x);
        if (orders.isEmpty()) {
            NOTE.put(m, "no orders today");
            return;
        }
        Map<String, Object> pol = policies.get(m) instanceof Map ? (Map<String, Object>) policies.get(m) : null;
        String barDay = today.replace("-", "");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (Ib ib = Ib.connect(Bridge.env("IB_HOST", "127.0.0.1"), Integer.parseInt(Bridge.env("IB_PORT", "4002")), 70 + Book.MARKETS.indexOf(m), 2)) {
            List<Broker.Fill> fills;
            try {
                fills = ib.fills();
            } catch (RuntimeException e) {
                fills = List.of();
            }
            for (Map<String, Object> od : orders.subList(0, Math.min(orders.size(), 6))) {
                String ticker = String.valueOf(od.get("ticker"));
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("time", now.format(DateTimeFormatter.ofPattern("HH:mm")));
                row.put("ticker", ticker);
                double sigma = num(od.get("sigma")), limit = num(od.get("limit"));
                Broker.Share s = Holds.shareOf(ticker, m);
                Broker.Share q = s == null ? null : ib.qualify(s);
                if (q == null) {
                    row.put("status", "not found at IBKR");
                    rows.add(row);
                    continue;
                }
                List<Bar> bars = new ArrayList<>();
                for (Bar b : ib.intradayBars(q, "", "1 D", 30)) if (b.time().trim().startsWith(barDay)) bars.add(b);
                if (bars.isEmpty() || !(sigma > 0)) {
                    row.put("status", bars.isEmpty() ? "no 5-minute bar yet" : "no volatility in the orders file");
                    rows.add(row);
                    continue;
                }
                double px = bars.get(bars.size() - 1).close();
                int c = bars.size() - 1;  // the bar's index in the day: the policy's check (5-minute checks: every = 1)
                double bought = 0, sold = 0, cost = 0;
                for (Broker.Fill f : fills) {
                    if (!f.share().symbol().equals(q.symbol()) || f.ref() == null || !f.ref().startsWith("ofm-")) continue;
                    if ("BOT".equals(f.side())) {
                        bought += f.shares();
                        cost += f.shares() * f.price();
                    } else if ("SLD".equals(f.side())) {
                        sold += f.shares();
                    }
                }
                row.put("price", px);
                row.put("bar", c);
                String kind;
                int bucket;
                boolean dayEntry = Boolean.TRUE.equals(od.get("entryDay"));
                if (bought > 0 && bought - sold > 1e-9) {
                    double entry = cost / bought, z = (px / entry - 1) / sigma;
                    kind = "pos";
                    bucket = bucket(z, buckets(pol, "zBuckets", new double[] {-1.5, -0.75, -0.25, 0.25, 0.75, 1.5}));
                    row.put("status", "bought");
                    row.put("entry", entry);
                    row.put("moveSigma", z);
                } else if (bought > 0) {
                    row.put("status", "sold today (take-profit, stop or close)");
                    rows.add(row);
                    continue;
                } else {
                    double g = limit > 0 ? (px / limit - 1) / sigma : Double.NaN;
                    kind = "wait";
                    bucket = bucket(g, buckets(pol, "gapBuckets", new double[] {0.25, 0.5, 1.0, 2.0}));
                    row.put("status", "not filled");
                    row.put("limit", limit);
                    row.put("gapSigma", g);
                }
                row.put("state", kind + " " + bucket);
                String base = kind.equals("pos") ? "hold" : "keep";
                if (pol == null) {
                    row.put("action", base);
                    row.put("policy", "none yet for " + m + " (recorded only)");
                } else {
                    int every = (int) Math.max(1, num(pol.get("every")));
                    String act = base;
                    if ((c + 1) % every == 0 && pol.get("states") instanceof List)
                        for (Object x : (List<Object>) pol.get("states")) {
                            Map<String, Object> st = x instanceof Map ? (Map<String, Object>) x : Map.of();
                            if (!kind.equals(st.get("kind")) || (int) num(st.get("check")) != c / every || (int) num(st.get("bucket")) != bucket) continue;
                            if (kind.equals("wait") && Boolean.TRUE.equals(st.get("dayEntry")) != dayEntry) continue;
                            act = String.valueOf(st.get("action"));
                        }
                    row.put("action", act);
                    row.put("policy", "learnt " + pol.getOrDefault("learnt", "?"));
                }
                rows.add(row);
            }
        }
        List<Map<String, Object>> day = DAY.compute(m, (k, v) -> v == null || v.isEmpty() || !today.equals(v.get(0).get("date")) ? new ArrayList<>() : v);
        try {
            Path f = Path.of(Auto.HISTORY, "shadow", m, today + ".jsonl");
            Files.createDirectories(f.getParent());
            StringBuilder sb = new StringBuilder();
            for (Map<String, Object> r : rows) {
                r.put("date", today);
                sb.append(Json.write(r)).append('\n');
            }
            Files.writeString(f, sb.toString(), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            Auto.err("shadow " + m + ": not written: " + e.getMessage());
        }
        synchronized (day) {
            day.addAll(rows);
            while (day.size() > 2000) day.remove(0);
        }
        NOTE.put(m, rows.size() + " orders checked at " + now.format(DateTimeFormatter.ofPattern("HH:mm z")));
    }

    /** The day's rows for GET /shadow (the file when the bridge restarted since). */
    static List<Object> rows(String m, String date) {
        List<Object> out = new ArrayList<>();
        try {
            Path f = Path.of(Auto.HISTORY, "shadow", m, date + ".jsonl");
            if (Files.exists(f)) for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) if (!l.isBlank()) out.add(JsonParse.object(l));
        } catch (IOException | RuntimeException e) {
            out.add(Map.of("status", "unreadable: " + e.getMessage()));
        }
        return out;
    }

    static double[] buckets(Map<String, Object> pol, String key, double[] fallback) {
        if (pol == null || !(pol.get(key) instanceof List<?> l) || l.isEmpty()) return fallback;
        double[] b = new double[l.size()];
        for (int i = 0; i < b.length; i++) b[i] = num(l.get(i));
        return b;
    }

    static int bucket(double x, double[] edges) {
        int n = 0;
        for (double v : edges) if (x > v) n++;
        return n;
    }

    static double num(Object o) {
        return o instanceof Number n ? n.doubleValue() : Double.NaN;
    }

    private Shadow() {
    }
}
