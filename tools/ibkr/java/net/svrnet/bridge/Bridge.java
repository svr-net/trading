// The broker bridge in Java, beside IB Gateway in one container (docker/ibkr/Dockerfile.ibkr). It serves the
// page and the Bridge monitor over HTTPS (the JDK's own server; the certificate reloaded when renewed), talks to
// IB Gateway through IBKR's official Java API (com.ib.client, GPL-3) and to IBC through IBC's command server, and
// runs each automatic market's day (Auto: the multi-day book, the close, the report, the recordings, the watchdog).
//
// GET /status (no key), /diag, /history, /trades?market=, /orders?market=, /chart?market=&ticker=&days=, /bars.csv?market=, /java;
// POST /place, /report, /close, /bars, /paths, /test-order {"market": UK | EU | US | APAC | INDEX}, /restart, /reset {"confirm": "reset"}, /ibc {"command": RESTART |
// RECONNECTACCOUNT | RECONNECTDATA}.
//
// Environment: BRIDGE_KEY, ALLOWED_ORIGIN, PORT (8765), BIND, TLS_P12 (the certificate as PKCS#12, password TLS_P12_PASSWORD; without it
// plain HTTP on 127.0.0.1 only), IB_HOST, IB_PORT (4002 paper), IBC_COMMAND_PORT (7462), EOD_TOKEN, CAPITAL, AUTO_MARKETS, HISTORY_DIR,
// ORDERS_DIR, BRIDGE_VERSION, and the schedule's (Auto).
package net.svrnet.bridge;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpsConfigurator;
import com.sun.net.httpserver.HttpsServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

public final class Bridge {
    static final String KEY = env("BRIDGE_KEY", "");
    static final String ORIGIN = env("ALLOWED_ORIGIN", "https://svr-net.github.io");
    static final String VERSION = env("BRIDGE_VERSION", "unknown");

    static String env(String name, String fallback) {
        String v = System.getenv(name);
        return v == null || v.isEmpty() ? fallback : v;
    }

    public static void main(String[] args) throws Exception {
        if (KEY.isEmpty()) {
            System.err.println("Set BRIDGE_KEY.");
            System.exit(2);
        }
        int port = Integer.parseInt(env("PORT", "8765"));
        String p12 = env("TLS_P12", "");
        HttpServer server;
        if (!p12.isEmpty()) {
            Tls tls = new Tls(Path.of(p12), env("TLS_P12_PASSWORD", "bridge").toCharArray());
            HttpsServer https = HttpsServer.create(new InetSocketAddress(env("BIND", "0.0.0.0"), port), 64);
            https.setHttpsConfigurator(new HttpsConfigurator(tls.context()));
            server = https;
        } else {  // no certificate: only this machine may reach it
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 64);
        }
        server.createContext("/", Bridge::handle);
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
        Auto.start();
        System.out.println("bridge " + VERSION + " on " + (p12.isEmpty() ? "http://127.0.0.1" : "https://*") + ":" + port + " for " + ORIGIN
                + "; IB Gateway at " + Auto.IB_HOST + ":" + Auto.IB_PORT + "; automatic: " + (Auto.AUTO.isEmpty() ? "off" : String.join(",", Auto.AUTO))
                + "; IBKR API " + Api.version());
    }

    static void handle(HttpExchange x) throws IOException {
        try {
            String path = x.getRequestURI().getPath();
            String method = x.getRequestMethod();
            if (method.equals("OPTIONS")) {
                preflight(x);
            } else if (method.equals("GET") && path.equals("/live")) {  // no key: the start script's liveness check (only the loops' lateness)
                Map<String, Long> late = Auto.late();
                reply(x, late.isEmpty() ? 200 : 503, Json.obj("live", late.isEmpty(), "late", late));
            } else if (method.equals("GET") && path.equals("/status")) {
                reply(x, 200, Json.write(Auto.status()));
            } else if (!allowed(x)) {
                return;
            } else if (method.equals("GET")) {
                get(x, path);
            } else if (method.equals("POST")) {
                post(x, path);
            } else {
                reply(x, 404, Json.obj("error", "not found"));
            }
        } catch (Exception e) {
            try {
                reply(x, 500, Json.obj("error", e.getClass().getSimpleName() + ": " + e.getMessage()));
            } catch (Exception ignored) {
                // the client has gone
            }
        } finally {
            x.close();
        }
    }

    // --- The bridge's own rules (as bridge.py): only the page's origin, and the key (a wrong one costs a second)
    static boolean allowed(HttpExchange x) throws IOException {
        String origin = x.getRequestHeaders().getFirst("Origin");
        if (origin != null && !origin.equals(ORIGIN)) {
            reply(x, 403, Json.obj("error", "origin not allowed"));
            return false;
        }
        String key = x.getRequestHeaders().getFirst("X-Bridge-Key");
        if (key == null || !MessageDigest.isEqual(key.getBytes(StandardCharsets.UTF_8), KEY.getBytes(StandardCharsets.UTF_8))) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            reply(x, 401, Json.obj("error", "key missing or wrong"));
            return false;
        }
        return true;
    }

    static void reply(HttpExchange x, int code, String json) throws IOException {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        if (ORIGIN.equals(x.getRequestHeaders().getFirst("Origin"))) {
            x.getResponseHeaders().set("Access-Control-Allow-Origin", ORIGIN);
            x.getResponseHeaders().set("Vary", "Origin");
        }
        x.getResponseHeaders().set("Content-Type", "application/json");
        x.sendResponseHeaders(code, body.length);
        try (OutputStream o = x.getResponseBody()) {
            o.write(body);
        }
    }

    static void preflight(HttpExchange x) throws IOException {  // the browser's preflight (with Chrome's private-network permission)
        if (ORIGIN.equals(x.getRequestHeaders().getFirst("Origin"))) {
            x.getResponseHeaders().set("Access-Control-Allow-Origin", ORIGIN);
            x.getResponseHeaders().set("Vary", "Origin");
        }
        x.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST");
        x.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, X-Bridge-Key");
        x.getResponseHeaders().set("Access-Control-Allow-Private-Network", "true");
        x.sendResponseHeaders(204, -1);
    }

    static void replyBytes(HttpExchange x, int code, byte[] body, String type) throws IOException {
        if (ORIGIN.equals(x.getRequestHeaders().getFirst("Origin"))) {
            x.getResponseHeaders().set("Access-Control-Allow-Origin", ORIGIN);
            x.getResponseHeaders().set("Vary", "Origin");
        }
        x.getResponseHeaders().set("Content-Type", type);
        x.sendResponseHeaders(code, body.length);
        try (OutputStream o = x.getResponseBody()) {
            o.write(body);
        }
    }

    static Map<String, String> query(HttpExchange x) {
        Map<String, String> q = new java.util.LinkedHashMap<>();
        String raw = x.getRequestURI().getRawQuery();
        for (String part : raw == null ? new String[0] : raw.split("&")) {
            int i = part.indexOf('=');
            if (i <= 0) continue;
            q.putIfAbsent(java.net.URLDecoder.decode(part.substring(0, i), StandardCharsets.UTF_8), java.net.URLDecoder.decode(part.substring(i + 1), StandardCharsets.UTF_8));
        }
        return q;
    }

    /** The request's market, or null after answering 400. */
    static String market(HttpExchange x, Object value) throws IOException {
        String m = value == null ? "" : value.toString().toUpperCase(Locale.ROOT);
        if (!Book.MARKETS.contains(m)) {
            reply(x, 400, Json.obj("error", "market must be UK, EU, US, APAC or INDEX"));
            return null;
        }
        return m;
    }

    static void get(HttpExchange x, String path) throws IOException {
        Map<String, String> q = query(x);
        switch (path) {
            case "/java" -> reply(x, 200, Json.obj("java", true, "version", VERSION, "api", Api.version()));
            case "/diag" -> reply(x, 200, Json.obj("checks", Auto.diag()));
            case "/history" -> reply(x, 200, Json.obj("days", Book.history()));
            case "/shadow" -> {  // shadow mode: the day's 5-minute checks and the learnt policy's actions (recorded only, keyed)
                String m = q.getOrDefault("market", "US").toUpperCase(), date = q.getOrDefault("date", Auto.today(Book.MARKETS.contains(m) ? m : "US"));
                if (!Book.MARKETS.contains(m) || !date.matches("\\d{4}-\\d{2}-\\d{2}")) {
                    reply(x, 400, Json.obj("error", "market one of " + Book.MARKETS + ", date yyyy-mm-dd"));
                    return;
                }
                reply(x, 200, Json.obj("market", m, "date", date, "rows", Shadow.rows(m, date), "policy", Shadow.policyNote, "hasPolicy", Shadow.policies.containsKey(m),
                        "note", Shadow.NOTE.getOrDefault(m, "")));
            }
            case "/watch" -> {  // the account watch's latest result (every 5 minutes)
                Object[] w = Auto.WATCH;
                if (w == null) {
                    reply(x, 202, Json.obj("ok", null, "items", List.of(), "note", "not checked yet: in a few minutes"));
                    return;
                }
                Object[] r = (Object[]) w[1];
                reply(x, 200, Json.obj("ok", r[0], "items", r[1], "age", Auto.ago((long) w[0])));
            }
            case "/portfolio" -> {  // the whole account at IBKR, from the background's latest read (keyed: positions and money)
                Object[] hit = Auto.PORTFOLIO;
                if (hit == null || Auto.ago((long) hit[0]) >= 55) Auto.readOnce("portfolio", Auto::readPortfolio);
                if (hit == null) {
                    reply(x, 202, Json.obj("error", Auto.PORTFOLIO_NOTE.isEmpty() ? "being read from IBKR: ask again in a minute" : Auto.PORTFOLIO_NOTE, "positions", List.of()));
                    return;
                }
                @SuppressWarnings("unchecked") Map<String, Object> out = new java.util.LinkedHashMap<>((Map<String, Object>) hit[1]);
                out.put("age", Auto.ago((long) hit[0]));
                out.put("note", Auto.PORTFOLIO_NOTE);
                reply(x, 200, Json.write(out));
            }
            case "/orders" -> orders(x);
            case "/trades" -> {  // the live state of today's trades and the model's holdings, from the background's latest read
                String m = market(x, q.get("market"));
                if (m == null) return;
                Object[] hit = Auto.TRADES.get(m);
                if (hit == null || Auto.ago((long) hit[0]) >= 55) Auto.readOnce("trades-" + m, () -> Auto.readTrades(m));
                if (hit == null) {
                    reply(x, 202, Json.obj("error", Auto.TRADES_NOTE.getOrDefault(m, "being read from IBKR: ask again in a minute"), "shares", List.of()));
                    return;
                }
                @SuppressWarnings("unchecked") Map<String, Object> out = new java.util.LinkedHashMap<>((Map<String, Object>) hit[1]);
                out.put("age", Auto.ago((long) hit[0]));
                out.put("note", Auto.TRADES_NOTE.getOrDefault(m, ""));
                reply(x, 200, Json.write(out));
            }
            case "/bars.csv" -> {  // the market's daily bars from IBKR (the kept file), for the page's Lab (keyed: prices)
                String m = market(x, q.get("market"));
                if (m == null) return;
                if (!Charts.ensureKept(m)) {
                    reply(x, 404, Json.obj("error", "no IBKR bars kept for " + m + " yet: they come with the next bars fetch"));
                    return;
                }
                replyBytes(x, 200, Files.readAllBytes(Charts.kept(m)), "text/csv");
            }
            case "/chart" -> {  // a share's daily bars, for the page's chart of an order (keyed: prices)
                String m = market(x, q.get("market"));
                if (m == null) return;
                String t = q.getOrDefault("ticker", "").strip();
                int days;
                try {
                    days = q.getOrDefault("days", "").isEmpty() ? 120 : Integer.parseInt(q.get("days"));
                } catch (NumberFormatException e) {
                    days = 120;
                }
                List<Map<String, Object>> bars = t.isEmpty() ? List.of() : Charts.bars(m, t, days);
                if (bars.isEmpty()) {
                    reply(x, 404, Json.obj("error", "no daily bars for " + (t.isEmpty() ? "?" : t) + " (" + m + ") from IBKR or the store"));
                    return;
                }
                reply(x, 200, Json.obj("market", m, "ticker", t, "bars", bars, "profile", Charts.profile(m, t)));
            }
            default -> reply(x, 404, Json.obj("error", "not found"));
        }
    }

    @SuppressWarnings("unchecked")
    static void post(HttpExchange x, String path) throws IOException {
        String raw = body(x);
        Map<String, Object> body;
        try {
            body = raw.isBlank() ? Map.of() : JsonParse.object(raw);
        } catch (RuntimeException e) {
            reply(x, 400, Json.obj("error", "bad JSON"));
            return;
        }
        if (path.equals("/ibc")) {
            ibc(x, String.valueOf(body.getOrDefault("command", "")));
            return;
        }
        if (path.equals("/reset")) {  // the records archived here and in the store (IBKR's bars kept): {"confirm": "reset"}
            if (!"reset".equals(body.get("confirm"))) {
                reply(x, 400, Json.obj("error", "send {\"confirm\": \"reset\"}: the history, the books' positions and the step marks are archived"));
                return;
            }
            if (!Auto.RUNNING.isEmpty()) {
                reply(x, 409, Json.obj("ok", false, "log", "steps are running (" + String.join(", ", Auto.RUNNING) + "): try again when they end"));
                return;
            }
            String note = Mirror.reset();
            Auto.log("-", "reset", true, note);
            reply(x, 200, Json.obj("ok", true, "log", note));
            return;
        }
        if (path.equals("/restart")) {  // IB Gateway restarted on demand (the Bridge monitor's button)
            String note = Auto.restartGateway();
            boolean ok = !note.startsWith("no IB Gateway");
            Auto.log("-", "gateway restart", ok, note);
            reply(x, 200, Json.obj("ok", ok, "log", note));
            return;
        }
        String m = market(x, body.get("market"));
        if (m == null) return;
        boolean dry = Boolean.TRUE.equals(body.get("dryRun"));
        switch (path) {
            case "/place" -> {
                if (dry) {  // what the book would place: nothing is sent to IBKR
                    Map<String, Object> book;
                    try {
                        book = Book.orders(m);
                    } catch (IOException e) {
                        reply(x, 502, Json.obj("error", "orders not available: " + e.getMessage()));
                        return;
                    }
                    StringBuilder log = new StringBuilder();
                    int n = 0;
                    for (Object o0 : (List<Object>) book.getOrDefault("orders", List.of())) {
                        if (!(o0 instanceof Map)) continue;
                        Map<String, Object> o = (Map<String, Object>) o0;
                        n++;
                        log.append("  ").append(o.get("ticker")).append(": buy ").append(o.get("quantity")).append(" at ")
                                .append(o.get("limit") instanceof Number ? Holds.g(((Number) o.get("limit")).doubleValue()) : "?")
                                .append("open".equals(o.get("entry")) ? " (the opening auction only)" : "").append('\n');
                    }
                    log.append(m).append(" dry run: ").append(n).append(" orders of the ").append("hold".equals(book.get("book")) ? "multi-day book" : "book")
                            .append(" from ").append(book.get("asOf")).append("; nothing sent to IBKR");
                    reply(x, 200, Json.obj("ok", true, "log", log.toString()));
                } else if (!Auto.liveIsHold(m)) {
                    reply(x, 409, Json.obj("ok", false, "log", m + ": the orders are not a multi-day book: the bridge places only multi-day books"));
                } else if (Auto.running(m, "hold-place")) {
                    reply(x, 409, Json.obj("ok", false, "log", m + ": the book is being placed already"));
                } else {
                    Auto.markDone(m, "place");
                    Auto.job(m, "hold-place", Auto::placeHolds, true);
                    reply(x, 202, Json.obj("ok", true, "log", m + ": placing the multi-day book again (a minute or two; buys already working are kept). Diagnostics shows the result."));
                }
            }
            case "/bars" -> {  // in the background: it takes over an hour
                if (Auto.BARS_RUNNING.contains(m)) {
                    reply(x, 409, Json.obj("ok", false, "log", m + " bars: already running"));
                    return;
                }
                Auto.inBackground(m, "bars", mk -> Auto.fetchBars(mk, false));
                reply(x, 202, Json.obj("ok", true, "log", m + ": fetching the pool's daily bars from IBKR (about an hour); the status shows when done"));
            }
            case "/paths" -> {
                if (Auto.PATHS_RUNNING.contains(m)) {
                    reply(x, 409, Json.obj("ok", false, "log", m + " paths: already recording"));
                    return;
                }
                Auto.inBackground(m, "paths", Auto::recordPaths);
                reply(x, 202, Json.obj("ok", true, "log", m + ": recording the candidates' prices until 20 minutes after the close"));
            }
            case "/intraday" -> {  // a backfill of the market's 5-minute bars from IBKR, in the background (days: 1 to 180)
                int days = body.get("days") instanceof Number ? Math.max(1, Math.min(180, ((Number) body.get("days")).intValue())) : 60;
                Auto.inBackground(m, "intraday", mk -> Auto.fetchIntraday(mk, days));
                reply(x, 202, Json.obj("ok", true, "log", m + ": fetching " + days + " sessions of 5-minute bars from IBKR in the background (about " + (days / 20 + 1) * 11 * Auto.INTRADAY_LINES / 60 + " minutes)"));
            }
            case "/test-order" -> testOrder(x, m);
            case "/report" -> {
                Auto.Done d = Auto.report(m);
                reply(x, d.ok() ? 200 : 500, Json.obj("ok", d.ok(), "log", d.summary()));
            }
            case "/liquidate" -> {  // the book's positions from earlier days sold now (their stops cancelled first)
                Auto.Done d = Auto.liquidateHolds(m);
                reply(x, d.ok() ? 200 : 500, Json.obj("ok", d.ok(), "log", d.summary()));
            }
            case "/close" -> {
                Auto.Done d = Auto.closeMarket(m);
                reply(x, d.ok() ? 200 : 500, Json.obj("ok", d.ok(), "log", d.summary()));
            }
            default -> reply(x, 404, Json.obj("error", "not found"));
        }
    }

    static String body(HttpExchange x) throws IOException {
        try (InputStream in = x.getRequestBody()) {
            return new String(in.readNBytes(1 << 16), StandardCharsets.UTF_8);
        }
    }

    // --- The day's orders (GET /orders?market=), as bridge.py
    static void orders(HttpExchange x) throws IOException {
        String q = x.getRequestURI().getRawQuery();
        String market = "";
        for (String part : q == null ? new String[0] : q.split("&")) {
            if (part.startsWith("market=")) market = java.net.URLDecoder.decode(part.substring(7), StandardCharsets.UTF_8).toUpperCase(Locale.ROOT);
        }
        if (!Book.MARKETS.contains(market)) {
            reply(x, 400, Json.obj("error", "market must be UK, EU, US, APAC or INDEX"));
            return;
        }
        Map<String, Object> book;
        try {
            Auto.fetchOrders(market);  // the model's own result for its latest day kept (record_check)
            book = Book.orders(market);
        } catch (Exception e) {
            reply(x, 502, Json.obj("error", "orders not available: " + e.getMessage()));
            return;
        }
        reply(x, 200, Json.write(book));
    }

    // --- IBC's command server: IB Gateway restarted or reconnected by IBC itself
    static final List<String> IBC_COMMANDS = List.of("RESTART", "RECONNECTACCOUNT", "RECONNECTDATA");

    static void ibc(HttpExchange x, String cmd) throws IOException {
        String command = cmd.toUpperCase(Locale.ROOT);
        if (!IBC_COMMANDS.contains(command)) {
            reply(x, 400, Json.obj("error", "command must be one of " + IBC_COMMANDS));
            return;
        }
        String answer = Ibc.send(command);
        boolean ok = answer.startsWith("OK");
        Auto.log("-", "ibc " + command.toLowerCase(Locale.ROOT), ok, "IBC: " + answer);
        reply(x, ok ? 200 : 502, Json.obj("ok", ok, "log", "IBC: " + answer));
    }

    // --- The test order (IBKR's official Java API)
    @SuppressWarnings("unchecked")
    static void testOrder(HttpExchange x, String market) throws IOException {
        Map<String, Object> r = new TestOrder(market).run();
        boolean ok = Boolean.TRUE.equals(r.get("accepted"));
        String msgs = String.join(" | ", (List<String>) r.getOrDefault("messages", List.of()));
        String summary = ok ? "test order accepted by IBKR (" + r.get("probe") + ", cancelled: " + r.get("cancelled") + ")"
                : "test order refused: " + r.getOrDefault("fate", "") + " [IBKR: " + (msgs.isEmpty() ? r.getOrDefault("error", "") : msgs) + "]";
        Auto.log(market, "test-order", ok, summary);
        reply(x, 200, Json.obj("ok", ok, "result", r, "log", summary));
    }

    static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        in.transferTo(b);
        return b.toByteArray();
    }

    static byte[] read(Path p) throws IOException {
        return Files.readAllBytes(p);
    }

    private Bridge() {
    }
}
