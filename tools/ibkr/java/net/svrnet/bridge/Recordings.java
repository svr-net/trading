package net.svrnet.bridge;

import com.ib.client.Bar;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

/**
 * What the bridge records from IBKR beside the orders (submit.py, ported): the candidates' opening auction, their
 * prices through the day (the paths), the pool's daily bars, the API check (diag) and the watchdog's ping. None
 * places anything.
 */
final class Recordings {
    static final Map<String, Object[]> CLOSE = Map.of("UK", new Object[] {"Europe/London", 16, 35}, "EU", new Object[] {"Europe/Paris", 17, 35},
            "US", new Object[] {"America/New_York", 16, 0}, "APAC", new Object[] {"Australia/Sydney", 16, 10}, "INDEX", new Object[] {"Europe/Berlin", 17, 35});
    static final Map<String, String> PROBE = Map.of("UK", "VOD", "US", "AAPL", "APAC", "BHP", "EU", "AIR", "INDEX", "EXS1");

    static int clientId(String market) {
        return 17 + 10 * Book.MARKETS.indexOf(market);
    }

    /** The shares a recording streams: the day's orders first, then the other candidates in their rank, at most `lines` (0: all). */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> streamed(Map<String, Object> book, int lines) {
        List<Map<String, Object>> orders = new ArrayList<>();
        for (Object o : (List<Object>) book.getOrDefault("orders", List.of())) if (o instanceof Map) orders.add((Map<String, Object>) o);
        List<Map<String, Object>> cands = new ArrayList<>();
        Object c0 = book.get("candidates");
        if (c0 instanceof List && !((List<Object>) c0).isEmpty()) {
            for (Object o : (List<Object>) c0) if (o instanceof Map) cands.add((Map<String, Object>) o);
        } else {
            cands.addAll(orders);
        }
        Set<Object> ordered = new HashSet<>();
        for (Map<String, Object> o : orders) ordered.add(o.get("ticker"));
        Set<Object> inCands = new HashSet<>();
        for (Map<String, Object> c : cands) inCands.add(c.get("ticker"));
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> c : cands) if (ordered.contains(c.get("ticker"))) out.add(c);
        for (Map<String, Object> o : orders) {
            if (inCands.contains(o.get("ticker"))) continue;
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("ticker", o.get("ticker"));
            m.put("close", o.get("close"));
            out.add(m);
        }
        for (Map<String, Object> c : cands) if (!ordered.contains(c.get("ticker"))) out.add(c);
        return lines > 0 && out.size() > lines ? new ArrayList<>(out.subList(0, lines)) : out;
    }

    static ZonedDateTime at(String market, Map<String, Object[]> table) {
        Object[] z = table.get(market);
        ZonedDateTime now = ZonedDateTime.now(Holds.clock.withZone(ZoneId.of((String) z[0])));
        return now.withHour((int) z[1]).withMinute((int) z[2]).withSecond(0).withNano(0);
    }

    static Double nul(Map<String, Double> f, String k) {
        return f.get(k);
    }

    /** The candidates' opening auction (generic tick 225: the indicative price, volume and imbalance), every 15 seconds until the open, then the open itself. */
    static Steps.Result auction(String market, Map<String, Object> book, int lines) {
        List<Map<String, Object>> cands = streamed(book, lines);
        ZonedDateTime opens = at(market, Holds.OPEN);
        ZoneId tz = opens.getZone();
        if (ZonedDateTime.now(tz).isAfter(opens.plusMinutes(5)))
            return new Steps.Result(1, List.of(), market + " opened at " + opens.format(DateTimeFormatter.ofPattern("HH:mm")) + ": too late to record its auction");
        try (Ib ib = Ib.connect(clientId(market) + 1)) {
            Map<String, Integer> ids = new LinkedHashMap<>();
            Map<String, Map<String, Object>> rows = new LinkedHashMap<>();
            for (Map<String, Object> c : cands) {
                String t = String.valueOf(c.get("ticker"));
                Broker.Share s = Holds.shareOf(t, market);
                Broker.Share q = s == null ? null : ib.qualify(s);
                if (q == null) continue;
                ids.put(t, ib.stream(q, "225", 1));
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("ticker", t);
                r.put("close", c.get("close"));
                r.put("score", c.get("score"));
                r.put("samples", new ArrayList<Object>());
                rows.put(t, r);
            }
            while (ZonedDateTime.now(tz).isBefore(opens)) {
                double left = (opens.toInstant().toEpochMilli() - System.currentTimeMillis()) / 1000.0;
                ib.sleep(Math.max(0.5, Math.min(15.0, left)));
                String now = ZonedDateTime.now(tz).format(DateTimeFormatter.ofPattern("HH:mm:ss"));
                for (Map.Entry<String, Integer> e : ids.entrySet()) {
                    Map<String, Double> f = ib.fields(e.getValue());
                    Map<String, Object> s = new LinkedHashMap<>();
                    s.put("at", now);
                    s.put("price", f.get("auctionPrice"));
                    s.put("volume", f.get("auctionVolume"));
                    s.put("imbalance", f.get("auctionImbalance"));
                    s.put("bid", f.get("bid"));
                    s.put("ask", f.get("ask"));
                    s.put("bidSize", f.get("bidSize"));
                    s.put("askSize", f.get("askSize"));
                    s.put("last", f.get("last"));
                    s.put("type", ib.dataType(e.getValue()));
                    @SuppressWarnings("unchecked") List<Object> l = (List<Object>) rows.get(e.getKey()).get("samples");
                    l.add(s);
                }
            }
            ib.sleep(120);  // the uncross and the first trades
            for (Map.Entry<String, Integer> e : ids.entrySet()) {
                Map<String, Double> f = ib.fields(e.getValue());
                Map<String, Object> r = rows.get(e.getKey());
                for (String k : List.of("open", "last", "volume", "high", "low")) r.put(k, f.get(k));
                ib.unstream(e.getValue());
            }
            int filled = 0;
            for (Map<String, Object> r : rows.values()) {
                @SuppressWarnings("unchecked") List<Map<String, Object>> l = (List<Map<String, Object>>) r.get("samples");
                for (Map<String, Object> s : l) if (s.get("price") != null && (double) s.get("price") != 0) {
                    filled++;
                    break;
                }
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("date", opens.toLocalDate().toString());
            out.put("market", market);
            out.put("asOf", book.get("asOf"));
            out.put("shares", new ArrayList<Object>(rows.values()));
            return new Steps.Result(0, List.of(market + " auction: " + rows.size() + " shares, an indicative price for " + filled), Json.write(out));
        } catch (RuntimeException e) {
            return new Steps.Result(1, List.of(), "auction: " + e.getMessage());
        }
    }

    /** The candidates' prices every 30 seconds from the open to 20 minutes after the close (delayed where not subscribed), appended to `out` as time,ticker,last,bid,ask,type. */
    static Steps.Result paths(String market, Map<String, Object> book, int lines, Path out) {
        List<Map<String, Object>> cands = streamed(book, lines);
        ZonedDateTime opens = at(market, Holds.OPEN);
        ZoneId tz = opens.getZone();
        ZonedDateTime ends = at(market, CLOSE).plusMinutes(20);
        if (ZonedDateTime.now(tz).isAfter(ends)) return new Steps.Result(1, List.of(), market + " closed at " + at(market, CLOSE).format(DateTimeFormatter.ofPattern("HH:mm")) + ": too late to record its day");
        try (Ib ib = Ib.connect(clientId(market) + 4)) {
            Map<String, Integer> ids = new LinkedHashMap<>();
            for (Map<String, Object> c : cands) {
                String t = String.valueOf(c.get("ticker"));
                Broker.Share s = Holds.shareOf(t, market);
                Broker.Share q = s == null ? null : ib.qualify(s);
                if (q != null) ids.put(t, ib.stream(q, "", 3));
            }
            int rows = 0;
            boolean fresh = !Files.exists(out) || Files.size(out) == 0;
            try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                if (fresh) w.write("time,ticker,last,bid,ask,type\r\n");
                while (ZonedDateTime.now(tz).isBefore(opens)) ib.sleep(Math.max(0.5, Math.min(30.0, (opens.toInstant().toEpochMilli() - System.currentTimeMillis()) / 1000.0)));
                while (ZonedDateTime.now(tz).isBefore(ends)) {
                    String now = ZonedDateTime.now(tz).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    for (Map.Entry<String, Integer> e : ids.entrySet()) {
                        Map<String, Double> f = ib.fields(e.getValue());
                        if (f.containsKey("last") || f.containsKey("bid")) {
                            w.write(now + "," + csv(e.getKey()) + "," + py(f.get("last")) + "," + py(f.get("bid")) + "," + py(f.get("ask")) + "," + ib.dataType(e.getValue()) + "\r\n");
                            rows++;
                        }
                    }
                    w.flush();
                    if (!ib.connected()) throw new IllegalStateException("IB Gateway closed the connection");
                    ib.sleep(30);
                }
            }
            for (int id : ids.values()) ib.unstream(id);
            return new Steps.Result(0, List.of(), market + " paths: " + ids.size() + " shares, " + rows + " prices");
        } catch (IOException | RuntimeException e) {
            return new Steps.Result(1, List.of(), "paths: " + e.getMessage());
        }
    }

    /** A number as Python's csv module writes a float ('' for none). */
    static String py(Double x) {
        if (x == null) return "";
        if (x == Math.rint(x) && Math.abs(x) < 1e16) return String.format(java.util.Locale.ROOT, "%.1f", x);
        double a = Math.abs(x);
        return a >= 1e7 || a < 1e-4 ? java.math.BigDecimal.valueOf(x).toPlainString() : Double.toString(x);
    }

    static String csv(String s) {
        return s.contains(",") || s.contains("\"") ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }

    /** The bars fetch's pause: while a step needs the gateway the fetch waits between two shares, and says so. */
    static final class Pause {
        private int n;
        private volatile boolean acked;

        synchronized void hold() {
            n++;
        }

        synchronized void release() {
            n = Math.max(0, n - 1);
            if (n == 0) acked = false;
        }

        synchronized boolean on() {
            return n > 0;
        }

        synchronized void clearIfFree() {
            if (n == 0) acked = false;
        }

        void ack(boolean v) {
            acked = v;
        }

        boolean acked() {
            return acked;
        }
    }

    /**
     * IBKR's daily bars (trades, regular hours) for the file's pool into `out` (date,ticker,open,high,low,close,volume). A share its exchange
     * gives no bars for is asked again through SMART (the exchange as its primary); when the first 3 of an exchange get none that way either,
     * the rest of it is not asked again. One share every 10.5 seconds (IBKR's pacing). The summary counts the shares without bars by IBKR's error.
     */
    @SuppressWarnings("unchecked")
    static Steps.Result bars(String market, Map<String, Object> book, String duration, Path out, Pause pause, Consumer<String> progress) {
        List<String> pool = new ArrayList<>();
        Object p0 = book.get("pool");
        if (p0 instanceof List && !((List<Object>) p0).isEmpty()) {
            for (Object x : (List<Object>) p0) pool.add(String.valueOf(x));
        } else {
            Object c0 = book.get("candidates");
            List<Object> src = c0 instanceof List ? (List<Object>) c0 : (List<Object>) book.getOrDefault("orders", List.of());
            for (Object x : src) if (x instanceof Map) pool.add(String.valueOf(((Map<String, Object>) x).get("ticker")));
        }
        while (pause.on()) {  // a step holds the gateway already: wait before even connecting
            pause.ack(true);
            sleep(2);
        }
        pause.ack(false);
        try (Ib ib = Ib.connect(clientId(market) + 2); BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            int got = 0, missing = 0, viaSmart = 0;
            Map<String, Integer> why = new LinkedHashMap<>();
            Map<String, Integer> smartTries = new HashMap<>();
            Set<String> smartWorks = new LinkedHashSet<>();
            double pausedFor = 0;
            w.write("date,ticker,open,high,low,close,volume\r\n");
            for (int k = 0; k < pool.size(); k++) {
                String t = pool.get(k);
                if (pause.on()) {
                    w.flush();
                    long t0 = System.nanoTime();
                    pause.ack(true);
                    while (pause.on()) ib.sleep(2);
                    pause.ack(false);
                    pausedFor += (System.nanoTime() - t0) / 1e9;
                }
                long start = System.nanoTime();
                Broker.Share s = Holds.shareOf(t, market);
                Broker.Share q = s == null ? null : ib.qualify(s);
                List<Bar> bars = List.of();
                if (q != null) {
                    bars = ib.bars(q, duration, 60);
                    String ex = q.exchange();
                    if (bars.isEmpty() && !ex.equals("SMART") && (smartWorks.contains(ex) || smartTries.getOrDefault(ex, 0) < 3)) {
                        sleep(Math.max(0, 10.5 - (System.nanoTime() - start) / 1e9));
                        Broker.Share alt = ib.qualify(new Broker.Share(0, q.symbol(), "SMART", ex, q.currency()));
                        if (alt != null) bars = ib.bars(alt, duration, 60);
                        if (!bars.isEmpty()) {
                            smartWorks.add(ex);
                            viaSmart++;
                        } else {
                            smartTries.merge(ex, 1, Integer::sum);
                        }
                    }
                    if (bars.isEmpty()) why.merge(ex + " " + (ib.barsError.isEmpty() ? "no error: no bars" : ib.barsError.length() > 90 ? ib.barsError.substring(0, 90) : ib.barsError), 1, Integer::sum);
                } else {
                    why.merge("not found at IBKR", 1, Integer::sum);
                }
                for (Bar b : bars) {
                    w.write(isoDay(b.time()) + "," + csv(t) + "," + py(b.open()) + "," + py(b.high()) + "," + py(b.low()) + "," + py(b.close()) + "," + vol(b) + "\r\n");
                }
                if (bars.isEmpty()) missing++;
                else got++;
                if ((k + 1) % 50 == 0) {
                    progress.accept("  " + (k + 1) + " of " + pool.size());
                    w.flush();
                }
                if (!ib.connected()) throw new IllegalStateException("IB Gateway closed the connection after " + (k + 1) + " of " + pool.size());
                if (k + 1 < pool.size()) sleep(Math.max(0, 10.5 - (System.nanoTime() - start) / 1e9));
            }
            List<Map.Entry<String, Integer>> top = new ArrayList<>(why.entrySet());
            top.sort((a, b) -> b.getValue() - a.getValue());
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Math.min(4, top.size()); i++) sb.append(i == 0 ? "" : "; ").append(top.get(i).getValue()).append(' ').append(top.get(i).getKey());
            return new Steps.Result(0, List.of(), market + " bars: " + got + " shares from IBKR (" + viaSmart + " through SMART), " + missing + " without bars"
                    + (pausedFor >= 30 ? "; paused " + Math.round(pausedFor / 60) + " minutes for the bridge's other steps" : "") + (sb.length() > 0 ? ": " + sb : ""));
        } catch (IOException | RuntimeException e) {
            return new Steps.Result(1, List.of(), "bars: " + e.getMessage());
        }
    }

    static String vol(Bar b) {
        double v = Ib.num(b.volume());
        return py(v);
    }

    /** A daily bar's date (IBKR's yyyyMMdd) as yyyy-MM-dd. */
    static String isoDay(String t) {
        String d = t == null ? "" : t.trim();
        if (d.length() >= 8 && d.substring(0, 8).chars().allMatch(Character::isDigit)) return d.substring(0, 4) + "-" + d.substring(4, 6) + "-" + d.substring(6, 8);
        return d;
    }

    /** The API, the account (paper or live) and whether the market's real-time and historical data come through, on one liquid share. */
    static Map<String, Object> diag(String market) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("market", market);
        Ib ib;
        try {
            ib = Ib.connect(Bridge.env("IB_HOST", "127.0.0.1"), Integer.parseInt(Bridge.env("IB_PORT", "4002")), clientId(market) + 3, 1);
        } catch (RuntimeException e) {
            out.put("api", false);
            out.put("error", e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage());
            return out;
        }
        try (ib) {
            ib.sleep(0.5);  // the accounts follow the connection
            List<String> accounts = ib.accounts();
            out.put("api", true);
            out.put("accounts", accounts.size());
            out.put("paper", accounts.isEmpty() ? null : accounts.stream().allMatch(a -> a.startsWith("D")));
            long t = ib.serverTime(20);
            out.put("serverTime", t == 0 ? null : Instant.ofEpochSecond(t).atOffset(ZoneOffset.UTC).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME));
            String[] v = Holds.VENUE.get(market);
            Broker.Share q = ib.qualify(new Broker.Share(0, PROBE.get(market), v[0], "", v[1]));
            if (q != null) {
                int id = ib.stream(q, "", 1);
                ib.sleep(4);
                Map<String, Double> f = ib.fields(id);
                out.put("probe", q.symbol());
                out.put("dataType", ib.dataType(id));
                for (String k : List.of("bid", "ask", "last", "close")) out.put(k, f.get(k));
                ib.unstream(id);
                List<Bar> bars = ib.bars(q, "5 D", 20);
                out.put("historyBars", bars.size());
                if (bars.isEmpty() && !ib.barsError.isEmpty()) out.put("historyError", ib.barsError);  // IBKR's reason (162, 354, ...)
            }
            // the account's positions, for the book-against-IBKR check: a slow answer (IBKR busy at a close, 24 Nov 16:21)
            // leaves that check out, never the whole IBKR check
            try {
                List<Map<String, Object>> held = new ArrayList<>();
                for (Broker.Position p : ib.positions()) {
                    if (p.qty() == 0) continue;
                    Map<String, Object> h = new LinkedHashMap<>();
                    h.put("symbol", p.share().symbol());
                    h.put("conId", (long) p.share().conId());
                    h.put("currency", p.share().currency());
                    held.add(h);
                }
                out.put("positions", held);
            } catch (RuntimeException e) {
                out.put("positionsError", e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage());
            }
            List<String> errs = new ArrayList<>();
            for (String n : ib.notes()) if (!n.startsWith("reader")) errs.add(n);
            out.put("errors", errs.subList(Math.max(0, errs.size() - 5), errs.size()));
        }
        return out;
    }

    /**
     * The watchdog's check: does IB Gateway answer, and does it still reach IBKR's servers? Its clock alone is not enough (a session taken over
     * by a login elsewhere still answers the clock, 6 Nov), so also one share's contract, within 20 seconds. Its own client (9).
     */
    static String ping() {
        try (Ib ib = Ib.connect(Bridge.env("IB_HOST", "127.0.0.1"), Integer.parseInt(Bridge.env("IB_PORT", "4002")), 9, 1)) {
            if (ib.serverTime(20) == 0) return "no answer: the clock";
            List<com.ib.client.ContractDetails> d;
            try {
                d = ib.details(new Broker.Share(0, "VOD", "SMART", "LSE", "GBP"), 20);
            } catch (RuntimeException e) {
                return "no answer: " + e.getMessage();
            }
            return d.isEmpty() ? "no answer from IBKR's servers: the contract lookup returned nothing" : "";
        } catch (RuntimeException e) {
            return "no answer: " + e.getMessage();
        }
    }

    static void sleep(double s) {
        try {
            Thread.sleep((long) (s * 1000));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private Recordings() {
    }
}
