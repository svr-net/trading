package net.svrnet.bridge;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The bridge's records that do not depend on the running scheduler (as bridge.py): GET /orders (the day's
 * orders file from svr-net/eod, each order's share count at the trade's amount) and GET /history (the orders
 * placed, the model's results and IBKR's fills, by day and market, from HISTORY_DIR/history.jsonl).
 */
final class Book {
    static final List<String> MARKETS = List.of("UK", "EU", "US", "APAC", "INDEX");
    static final double CAPITAL = Double.parseDouble(Bridge.env("CAPITAL", "5000"));
    static final String CACHE = Bridge.env("ORDERS_DIR", "/tmp/ofm-orders");
    static final String HISTORY = Bridge.env("HISTORY_DIR", Bridge.env("ORDERS_DIR", "/tmp/ofm-orders"));

    /** The day's orders file: a local one (ORDERS_FILE_<MARKET>), else svr-net/eod's through GitHub's contents API. */
    static Path fetch(String market) throws IOException {
        String local = System.getenv("ORDERS_FILE_" + market);
        if (local != null && !local.isEmpty()) return Path.of(local);
        String token = Bridge.env("EOD_TOKEN", Bridge.env("EOD_READ_TOKEN", ""));
        if (token.isEmpty()) throw new IOException("no EOD_TOKEN in the bridge (docker/ibkr/nas/.env)");
        HttpURLConnection c = (HttpURLConnection) new URL("https://api.github.com/repos/svr-net/eod/contents/orders/" + market + ".json").openConnection();
        c.setConnectTimeout(20_000);
        c.setReadTimeout(20_000);
        c.setRequestProperty("Authorization", "Bearer " + token);
        c.setRequestProperty("Accept", "application/vnd.github.raw");
        c.setRequestProperty("User-Agent", "ofm-bridge");
        c.setRequestProperty("Cache-Control", "no-cache");
        int code = c.getResponseCode();
        if (code == 401) throw new IOException("EOD_TOKEN rejected by GitHub (expired or revoked): put the current token in .env");
        if (code == 404) throw new IOException("orders/" + market + ".json not readable: the EOD_TOKEN has no access to svr-net/eod "
                + "(Repository access, Contents read), or the " + market + " job has not written it yet");
        if (code != 200) throw new IOException("GitHub answered " + code);
        Files.createDirectories(Path.of(CACHE));
        Path out = Path.of(CACHE, market + ".json");
        Path tmp = Path.of(CACHE, market + ".json.java-tmp");
        try (InputStream in = c.getInputStream()) {
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        return out;
    }

    /** GET /orders?market=: the orders file with each order's quantity (UK limits are in pence), the market and the capital. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> orders(String market) throws IOException {
        Map<String, Object> book = JsonParse.object(Files.readString(fetch(market), StandardCharsets.UTF_8));
        Object os = book.get("orders");
        if (os instanceof List) {
            for (Object x : (List<Object>) os) {
                if (!(x instanceof Map)) continue;
                Map<String, Object> o = (Map<String, Object>) x;
                double limit = num(o.get("limit"));
                double share = market.equals("UK") ? limit / 100.0 : limit;
                double capital = o.get("capital") instanceof Number && num(o.get("capital")) != 0 ? num(o.get("capital")) : CAPITAL;
                o.put("quantity", share > 0 ? (long) Math.floor(capital / share) : 0L);
            }
        }
        book.put("market", market);
        book.put("capital", CAPITAL);
        return book;
    }

    /** GET /history: by day and market (newest first): the orders placed, IBKR's fills (the day's latest report) and the model's own result. */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> history() throws IOException {
        Path f = Path.of(HISTORY, "history.jsonl");
        Map<String, Map<String, Object>> days = new LinkedHashMap<>();
        if (Files.exists(f)) {
            for (String line : Files.readAllLines(f, StandardCharsets.UTF_8)) {
                Map<String, Object> r;
                try {
                    r = JsonParse.object(line);
                } catch (IllegalArgumentException e) {
                    continue;
                }
                Object date = r.get("date"), market = r.get("market"), kind = r.get("kind");
                if (date == null || market == null) continue;
                Map<String, Object> d = days.computeIfAbsent(date + "\u0000" + market, k -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("date", date);
                    m.put("market", market);
                    m.put("orders", new ArrayList<>());
                    m.put("trades", new ArrayList<>());
                    m.put("model", new ArrayList<>());
                    return m;
                });
                if ("placed".equals(kind)) d.put("orders", r.getOrDefault("orders", new ArrayList<>()));
                else if ("result".equals(kind)) d.put("trades", r.getOrDefault("trades", new ArrayList<>()));
                else if ("model".equals(kind)) d.put("model", r.getOrDefault("trades", new ArrayList<>()));
            }
        }
        List<Map<String, Object>> out = new ArrayList<>(days.values());
        out.sort(Comparator.<Map<String, Object>, String>comparing(d -> String.valueOf(d.get("date")))
                .thenComparing(d -> String.valueOf(d.get("market"))).reversed());
        return out;
    }

    static double num(Object v) {
        return v instanceof Number ? ((Number) v).doubleValue() : 0.0;
    }

    private Book() {
    }
}
