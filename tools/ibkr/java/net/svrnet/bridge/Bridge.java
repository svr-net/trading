// The broker bridge in Java, beside IB Gateway in one container (docker/ibkr/Dockerfile.ibkr). It serves the
// page and the Bridge monitor over HTTPS (the JDK's own server; the certificate reloaded when renewed), talks to
// IB Gateway through IBKR's official Java API (com.ib.client, GPL-3) and to IBC through IBC's command server.
// Being moved from Python endpoint by endpoint: the endpoints here are answered here, every other request is
// passed unchanged to the Python bridge (tools/ibkr/bridge.py) on 127.0.0.1, so nothing stops working meanwhile.
//
// Here: GET /java (this bridge's version and the API's), POST /ibc {"command": RESTART | RECONNECTACCOUNT |
// RECONNECTDATA} (IBC's own commands for IB Gateway), POST /test-order {"market": UK | EU | US | APAC | INDEX}
// (does IBKR accept an order now? a 1-share limit buy at half the probe share's close, cancelled at once).
//
// Environment: BRIDGE_KEY, ALLOWED_ORIGIN, PORT (8765), TLS_P12 (the certificate as PKCS#12, password
// TLS_P12_PASSWORD; without it plain HTTP on 127.0.0.1 only), PYTHON_BRIDGE (http://127.0.0.1:8766), IB_HOST,
// IB_PORT (4002 paper), IBC_COMMAND_PORT (7462), BRIDGE_VERSION.
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
    static final String PYTHON = env("PYTHON_BRIDGE", "http://127.0.0.1:8766");
    static final String VERSION = env("BRIDGE_VERSION", "unknown");

    static String env(String name, String fallback) {
        String v = System.getenv(name);
        return v == null || v.isEmpty() ? fallback : v;
    }

    public static void main(String[] args) throws Exception {
        // The page's Origin is passed on to the Python bridge (its CORS answer): Java drops it unless allowed.
        System.setProperty("sun.net.http.allowRestrictedHeaders", "true");
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
        System.out.println("java bridge " + VERSION + " on " + (p12.isEmpty() ? "http://127.0.0.1" : "https://*") + ":" + port
                + "; Python bridge at " + PYTHON + "; IBKR API " + Api.version());
    }

    static void handle(HttpExchange x) throws IOException {
        try {
            String path = x.getRequestURI().getPath();
            String method = x.getRequestMethod();
            if (method.equals("GET") && path.equals("/java")) {
                if (allowed(x)) reply(x, 200, Json.obj("java", true, "version", VERSION, "api", Api.version()));
            } else if (method.equals("POST") && path.equals("/ibc")) {
                if (allowed(x)) ibc(x);
            } else if (method.equals("POST") && path.equals("/test-order")) {
                if (allowed(x)) testOrder(x);
            } else {
                proxy(x);
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

    static String body(HttpExchange x) throws IOException {
        try (InputStream in = x.getRequestBody()) {
            return new String(in.readNBytes(1 << 16), StandardCharsets.UTF_8);
        }
    }

    // --- IBC's command server: IB Gateway restarted or reconnected by IBC itself
    static final List<String> IBC_COMMANDS = List.of("RESTART", "RECONNECTACCOUNT", "RECONNECTDATA");

    static void ibc(HttpExchange x) throws IOException {
        String command = Json.field(body(x), "command").toUpperCase(Locale.ROOT);
        if (!IBC_COMMANDS.contains(command)) {
            reply(x, 400, Json.obj("error", "command must be one of " + IBC_COMMANDS));
            return;
        }
        String answer = Ibc.send(command);
        boolean ok = answer.startsWith("OK");
        Python.action("-", "ibc " + command.toLowerCase(Locale.ROOT), ok, "IBC: " + answer);
        reply(x, ok ? 200 : 502, Json.obj("ok", ok, "log", "IBC: " + answer));
    }

    // --- The test order (IBKR's official Java API)
    static void testOrder(HttpExchange x) throws IOException {
        String market = Json.field(body(x), "market").toUpperCase(Locale.ROOT);
        if (!TestOrder.PROBE.containsKey(market)) {
            reply(x, 400, Json.obj("error", "market must be one of " + TestOrder.PROBE.keySet()));
            return;
        }
        Map<String, Object> r = new TestOrder(market).run();
        boolean ok = Boolean.TRUE.equals(r.get("accepted"));
        String msgs = String.join(" | ", (List<String>) r.getOrDefault("messages", List.of()));
        String summary = ok ? "test order accepted by IBKR (" + r.get("probe") + ", cancelled: " + r.get("cancelled") + "; Java)"
                : "test order refused: " + r.getOrDefault("fate", "") + " [IBKR: " + (msgs.isEmpty() ? r.getOrDefault("error", "") : msgs) + "]";
        Python.action(market, "test-order", ok, summary);
        reply(x, 200, Json.obj("ok", ok, "result", r, "log", summary));
    }

    // --- Everything else: passed unchanged to the Python bridge (until it is moved here)
    static void proxy(HttpExchange x) throws IOException {
        URL url = new URL(PYTHON + x.getRequestURI().toString());
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setRequestMethod(x.getRequestMethod());
        c.setConnectTimeout(5000);
        c.setReadTimeout(900_000);  // a placing can take minutes
        c.setInstanceFollowRedirects(false);
        for (String h : List.of("Origin", "X-Bridge-Key", "Content-Type", "Access-Control-Request-Method",
                "Access-Control-Request-Headers", "Access-Control-Request-Private-Network")) {
            String v = x.getRequestHeaders().getFirst(h);
            if (v != null) c.setRequestProperty(h, v);
        }
        byte[] in;
        try (InputStream s = x.getRequestBody()) {
            in = s.readAllBytes();
        }
        if (in.length > 0 || x.getRequestMethod().equals("POST")) {
            c.setDoOutput(true);
            try (OutputStream o = c.getOutputStream()) {
                o.write(in);
            }
        }
        int code;
        try {
            code = c.getResponseCode();
        } catch (IOException e) {
            reply(x, 502, Json.obj("error", "the Python bridge does not answer: " + e.getMessage()));
            return;
        }
        for (Map.Entry<String, List<String>> h : c.getHeaderFields().entrySet()) {
            String k = h.getKey();
            if (k == null) continue;
            String lk = k.toLowerCase(Locale.ROOT);
            if (lk.equals("content-length") || lk.equals("transfer-encoding") || lk.equals("connection") || lk.equals("server") || lk.equals("date")) continue;
            x.getResponseHeaders().put(k, h.getValue());
        }
        InputStream s = code >= 400 ? c.getErrorStream() : c.getInputStream();
        byte[] out = s == null ? new byte[0] : s.readAllBytes();
        x.sendResponseHeaders(code, out.length == 0 ? -1 : out.length);
        if (out.length > 0) {
            try (OutputStream o = x.getResponseBody()) {
                o.write(out);
            }
        }
    }

    /** The Python bridge's record of the latest automatic actions (/diag, the monitor), kept until it moves here. */
    static final class Python {
        static void action(String market, String action, boolean ok, String summary) {
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(PYTHON + "/internal/action").openConnection();
                c.setRequestMethod("POST");
                c.setConnectTimeout(3000);
                c.setReadTimeout(5000);
                c.setDoOutput(true);
                c.setRequestProperty("X-Bridge-Key", KEY);
                c.setRequestProperty("Content-Type", "application/json");
                try (OutputStream o = c.getOutputStream()) {
                    o.write(Json.obj("market", market, "action", action, "ok", ok, "summary", summary).getBytes(StandardCharsets.UTF_8));
                }
                c.getResponseCode();
            } catch (IOException e) {
                System.err.println("java bridge: the action not recorded in the Python bridge: " + e.getMessage());
            }
        }
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
