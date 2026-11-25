package net.svrnet.bridge;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Socket;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * The bridge's automatic day (bridge.py, ported): each automatic market's steps that should be done by now, checked every 30 seconds and
 * done when missing, so a restart, a redeploy or a failed step is caught up rather than lost; the watchdog (IB Gateway's API, restarted
 * through IBC when it hangs); the background reads (the page's trades, the IBKR check); and the records the endpoints answer from.
 *
 * Every market's live orders are a multi-day book ("book": "hold", apps/ofm_strategy): the hold book places them (hold-place), protects its
 * buy-backs through the session (hold-protect) and offers them in the closing auction (hold-close). The model's other shares still held from
 * an earlier day are sold at the open (leftovers), the day's own orders closed (close), the check after the close that none is still held
 * (flat), the report of the day's fills, the opening auction and the day's paths recorded, and IBKR's daily bars fetched.
 */
final class Auto {
    static final List<String> MARKETS = Book.MARKETS;
    static final Map<String, String> ZONE = Map.of("UK", "Europe/London", "EU", "Europe/Paris", "US", "America/New_York", "APAC", "Australia/Sydney", "INDEX", "Europe/Berlin");
    static final List<String> AUTO = new ArrayList<>();
    static final Map<String, String[]> TIMES = new LinkedHashMap<>();
    static final Map<String, String> REPORT = new LinkedHashMap<>();
    static final Map<String, String> AUCTION = new LinkedHashMap<>();
    static final Map<String, String> HOLD_BOOKS = new LinkedHashMap<>();
    static final String BARS_AT = env("AUTO_BARS", "06:00", true);
    static final boolean BARS_DAILY = !env("BARS_DAILY", "1", false).equals("0");
    static final boolean PATHS = !env("PATHS", "1", false).equals("0");
    static final int LINES_BUDGET = Integer.parseInt(env("MARKET_LINES", "90", false));
    static final String HISTORY = Book.HISTORY;
    static final String CACHE = Book.CACHE;
    static final double CAPITAL = Book.CAPITAL;
    static final String IB_HOST = Bridge.env("IB_HOST", "127.0.0.1");
    static final int IB_PORT = Integer.parseInt(Bridge.env("IB_PORT", "4002"));
    static final String BUYS_URL = Bridge.env("BUYS_URL", "https://raw.githubusercontent.com/svr-net/trading/main/config/nas-buys.json");
    static final ZonedDateTime STARTED = ZonedDateTime.now();
    static final String UPDATE_NOTE = "one container: update it with docker/ibkr/ibkr-nas.sh";

    static {
        for (String m : Bridge.env("AUTO_MARKETS", "").split(",")) {
            String u = m.trim().toUpperCase();
            if (MARKETS.contains(u) && !AUTO.contains(u)) AUTO.add(u);
        }
        String[][] t = {{"UK", "07:45", "16:20"}, {"US", "09:20", "15:40"}, {"EU", "08:45", "17:20"}, {"APAC", "09:45", "15:50"}, {"INDEX", "08:46", "17:21"}};
        for (String[] x : t) TIMES.put(x[0], new String[] {env("AUTO_" + x[0] + "_PLACE", x[1], false), env("AUTO_" + x[0] + "_CLOSE", x[2], false)});
        String[][] r = {{"INDEX", "17:55"}, {"UK", "16:45"}, {"US", "16:10"}, {"APAC", "16:30"}, {"EU", "17:50"}};
        for (String[] x : r) REPORT.put(x[0], env("AUTO_" + x[0] + "_REPORT", x[1], false));
        if (!env("AUCTION", "1", false).equals("0")) {
            String[][] a = {{"INDEX", "08:53"}, {"UK", "07:52"}, {"US", "09:24"}, {"APAC", "09:52"}, {"EU", "08:52"}};
            for (String[] x : a) AUCTION.put(x[0], env("AUTO_" + x[0] + "_AUCTION", x[1], false));
        }
        for (String x : env("HOLD_BOOKS", "UK:hold-up", true).split(",")) {
            int i = x.indexOf(':');
            if (i > 0) HOLD_BOOKS.put(x.substring(0, i), x.substring(i + 1));
        }
    }

    /** An environment variable; `emptyOk`: set but empty means empty (Python's os.environ.get), not the default. */
    static String env(String name, String fallback, boolean emptyOk) {
        String v = System.getenv(name);
        if (v == null) return fallback;
        return v.isEmpty() && !emptyOk ? fallback : v;
    }

    // --- the records
    record Action(String when, String market, String action, boolean ok, String summary) { }

    static final List<Action> LAST = Collections.synchronizedList(new ArrayList<>());
    static final Map<String, Object[]> TRADES = new ConcurrentHashMap<>();  // market -> {when (nanos), the live state}
    static final Map<String, String> TRADES_NOTE = new ConcurrentHashMap<>();
    static volatile Long ibdiagAt;
    static volatile Map<String, Object> ibdiag;
    static final Set<String> READING = ConcurrentHashMap.newKeySet();
    static final Set<String> RUNNING = ConcurrentHashMap.newKeySet();  // "MARKET action" jobs in flight
    static final Map<String, long[]> ATTEMPTS = new ConcurrentHashMap<>();  // "MARKET action day" -> {tries, last try (nanos)}
    static final Map<String, Object> GATEWAY = Collections.synchronizedMap(new LinkedHashMap<>());
    static final Set<String> BARS_RUNNING = ConcurrentHashMap.newKeySet();
    static final Set<String> PATHS_RUNNING = ConcurrentHashMap.newKeySet();
    static final Set<String> PROTECT_RUNNING = ConcurrentHashMap.newKeySet();
    static final AtomicInteger BACKGROUND = new AtomicInteger();
    static final Recordings.Pause PAUSE = new Recordings.Pause();
    static final Map<String, Integer> LINES_USED = new LinkedHashMap<>();
    static final Map<String, Object[]> LIVE_HOLD = new ConcurrentHashMap<>();
    static final Object START_LOCK = new Object();
    static volatile Boolean buysOn;
    static volatile long buysChecked = Long.MIN_VALUE / 2;
    static volatile String buysNote = "not read yet";
    static volatile long restartedForRefusal = Long.MIN_VALUE / 2;
    static volatile long restartedLocal = Long.MIN_VALUE / 2;

    static {
        GATEWAY.put("ok", null);
        GATEWAY.put("fails", 0);
        GATEWAY.put("checked", null);
        GATEWAY.put("restarts", 0);
        GATEWAY.put("restarted", null);
        GATEWAY.put("note", "");
    }

    static String stamp(String zone) {
        return ZonedDateTime.now(ZoneId.of(zone)).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z"));
    }

    static void log(String market, String action, boolean ok, String summary) {
        String zone = ZONE.getOrDefault(market, "UTC");
        synchronized (LAST) {
            LAST.add(0, new Action(stamp(zone), market, action, ok, summary == null ? "" : summary));
            while (LAST.size() > 10) LAST.remove(LAST.size() - 1);
        }
    }

    static List<Action> last() {
        synchronized (LAST) {
            return new ArrayList<>(LAST);
        }
    }

    static ZonedDateTime now(String m) {
        return ZonedDateTime.now(ZoneId.of(ZONE.get(m)));
    }

    static String today(String m) {
        return now(m).toLocalDate().toString();
    }

    /**
     * The market's own date, for its steps' order references and fills. Not the NAS's: Sydney's morning is still the
     * day before in London (24 Nov: APAC's buys were tagged 23 Nov and its close, on 24 Nov, found none of them).
     */
    static LocalDate marketDay(String m) {
        return now(m).toLocalDate();
    }

    static ZonedDateTime at(ZonedDateTime now, String hhmm, int minutes) {
        String[] p = hhmm.split(":");
        return now.withHour(Integer.parseInt(p[0])).withMinute(Integer.parseInt(p[1])).withSecond(0).withNano(0).plusMinutes(minutes);
    }

    static boolean within(ZonedDateTime now, String hhmm, int from, int to) {
        return !now.isBefore(at(now, hhmm, from)) && now.isBefore(at(now, hhmm, to));
    }

    static boolean between(ZonedDateTime now, ZonedDateTime a, ZonedDateTime b) {
        return !now.isBefore(a) && now.isBefore(b);
    }

    static void err(String s) {
        System.err.println("bridge: " + s);
    }

    // --- files
    static Path state(String m, String action, String day) {
        return Path.of(HISTORY, "state", day, m + "-" + action);
    }

    static boolean isDone(String m, String action) {
        return Files.exists(state(m, action, today(m)));
    }

    /** A step of today done, kept in the history folder (it outlives restarts and redeploys). */
    static void markDone(String m, String action) {
        try {
            Path f = state(m, action, today(m));
            Files.createDirectories(f.getParent());
            if (!Files.exists(f)) Files.createFile(f);
            Mirror.changed();
        } catch (IOException e) {
            err("mark done " + m + " " + action + ": " + e.getMessage());
        }
    }

    static List<String> historyLines() {
        try {
            return Files.readAllLines(Path.of(HISTORY, "history.jsonl"), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return List.of();
        }
    }

    static Map<String, Object> parse(String line) {
        try {
            return JsonParse.object(line);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** Today's steps already in the history (placed, reported) marked done at start: a bridge upgraded mid-day never reports a day twice. */
    static void seedDone() {
        List<String> lines = historyLines();
        for (String line : lines.subList(Math.max(0, lines.size() - 200), lines.size())) {
            Map<String, Object> r = parse(line);
            if (r == null) continue;
            String m = String.valueOf(r.get("market"));
            String step = "placed".equals(r.get("kind")) ? "place" : "result".equals(r.get("kind")) ? "report" : null;
            if (ZONE.containsKey(m) && step != null && today(m).equals(r.get("date")) && !isDone(m, step)) markDone(m, step);
        }
    }

    static synchronized void historyAdd(Map<String, Object> record) throws IOException {
        Files.createDirectories(Path.of(HISTORY));
        Files.writeString(Path.of(HISTORY, "history.jsonl"), Json.write(record) + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        Mirror.changed();
    }

    /** The day's orders file (Book.fetch), the model's own result for its latest day kept once in the history. */
    static Path fetchOrders(String m) throws IOException {
        Path p = Book.fetch(m);
        if (System.getenv("ORDERS_FILE_" + m) == null) recordCheck(m, p);
        return p;
    }

    /** A diagnostics read of GitHub (the store's write access, a market's orders file), kept 5 minutes: GET /diag answers
     *  at once while GitHub is slow (23 Nov: 102 s), and a failure is not kept. */
    interface IoCall<T> { T call() throws IOException; }
    private static final Map<String, Object[]> DIAG_MEMO = new java.util.concurrent.ConcurrentHashMap<>();
    @SuppressWarnings("unchecked")
    static <T> T diagMemo(String key, IoCall<T> f) throws IOException {
        final Object[] hit = DIAG_MEMO.get(key);
        if (hit != null && System.nanoTime() - (long) hit[0] < 300_000_000_000L) return (T) hit[1];
        final T v = f.call();
        DIAG_MEMO.put(key, new Object[] {System.nanoTime(), v});
        return v;
    }

    static Map<String, Object> orders(String m) throws IOException {
        return JsonParse.object(Files.readString(fetchOrders(m), StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    static void recordCheck(String m, Path path) {
        try {
            Object c = JsonParse.object(Files.readString(path, StandardCharsets.UTF_8)).get("check");
            if (!(c instanceof Map)) return;
            Map<String, Object> check = (Map<String, Object>) c;
            Object d = check.get("date"), t = check.get("trades");
            if (d == null || "".equals(d) || !(t instanceof List) || ((List<Object>) t).isEmpty()) return;
            Path mark = Path.of(HISTORY, "model-" + m + "-" + d);
            if (Files.exists(mark)) return;
            Map<String, Object> rec = new LinkedHashMap<>();
            rec.put("kind", "model");
            rec.put("date", d);
            rec.put("market", m);
            rec.put("trades", t);
            historyAdd(rec);
            Files.createFile(mark);
            Mirror.changed();
        } catch (IOException | RuntimeException e) {
            err("model check " + m + ": " + e.getMessage());
        }
    }

    // --- IBKR refusing the orders themselves (2139: 16-19 Nov, every buy dead after IB Gateway restarted)
    static boolean refused(String out) {
        return out != null && (out.contains("2139:") || out.contains("paper-trading has not been acknowledged"));
    }

    static void restartForRefusal(String m, String action) {
        if (System.nanoTime() - restartedForRefusal < 20L * 60 * 1_000_000_000L) return;
        restartedForRefusal = System.nanoTime();
        String note = restartGateway();
        GATEWAY.put("restarts", (int) GATEWAY.get("restarts") + 1);
        GATEWAY.put("restarted", stamp(ZoneId.systemDefault().getId()));
        GATEWAY.put("note", "IBKR refused the " + m + " " + action + " orders (2139): " + note);
        err("IBKR refused the " + m + " " + action + " orders (2139): " + note);
        Recordings.sleep(240);
    }

    /** IB Gateway restarted by IBC (its command server), else its processes stopped (the start script starts it again); at most once in 5 minutes. */
    static String restartGateway() {
        if (System.nanoTime() - restartedLocal < 300L * 1_000_000_000L) return "restarted less than 5 minutes ago: not again yet";
        restartedLocal = System.nanoTime();
        String answer = Ibc.send("RESTART");
        if (answer.startsWith("OK")) return "IB Gateway restarted by IBC (" + answer + "); logged in again within about 2 minutes";
        int killed = 0;
        for (String pattern : List.of("ibcalpha.ibc", "ibcstart.sh")) {
            try {
                if (new ProcessBuilder("pkill", "-TERM", "-f", pattern).redirectErrorStream(true).start().waitFor() == 0) killed++;
            } catch (IOException e) {
                err("pkill: " + e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return killed > 0 ? "IB Gateway stopped; the container starts it again (logged in within about 2 minutes)" : "no IB Gateway process found to restart";
    }

    static boolean gatewayUp() {
        try (Socket s = new Socket()) {
            s.connect(new InetSocketAddress(IB_HOST, IB_PORT), 2000);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    // --- the steps: a daily-bars fetch (hundreds of requests, an hour or more) slows the gateway's other answers, so every step pauses it first
    interface Step {
        Steps.Result run() throws Exception;
    }

    static Steps.Result step(Step s) {
        PAUSE.hold();
        try {
            if (!BARS_RUNNING.isEmpty()) {
                int i = 0;
                while (i++ < 150 && !PAUSE.acked()) Recordings.sleep(1);
                if (!PAUSE.acked()) err("the bars fetch did not confirm its pause in time; the step runs anyway");
            }
            return s.run();
        } catch (Exception e) {
            return new Steps.Result(1, List.of(), e.getClass().getSimpleName() + ": " + e.getMessage());
        } finally {
            PAUSE.release();
        }
    }

    record Done(boolean ok, String summary) { }

    static void inBackground(String m, String action, Function<String, Done> job) {
        Thread t = new Thread(() -> {
            BACKGROUND.incrementAndGet();
            Done d;
            try {
                d = job.apply(m);
            } catch (Throwable e) {  // NOPMD: a failed step is logged, never lost silently
                d = new Done(false, e.getClass().getSimpleName() + ": " + e.getMessage());
            } finally {
                BACKGROUND.decrementAndGet();
            }
            log(m, action, d.ok(), d.summary());
            err(action + " " + m + ": " + (d.ok() ? "ok" : "FAILED") + " " + d.summary());
        }, action + "-" + m);
        t.setDaemon(true);
        t.start();
    }

    /** Runs the step in the background once at a time; marks it done when it succeeds. */
    static void job(String m, String action, Function<String, Done> fn, boolean doneOnOk) {
        String key = m + " " + action;
        if (!RUNNING.add(key)) return;
        String day = today(m);
        long[] a = ATTEMPTS.getOrDefault(key + " " + day, new long[] {0, 0});
        ATTEMPTS.put(key + " " + day, new long[] {a[0] + 1, System.nanoTime()});
        inBackground(m, action, mk -> {
            try {
                Done d = fn.apply(mk);
                if (d.ok() && doneOnOk) markDone(mk, action);
                if (!d.ok() && refused(d.summary())) restartForRefusal(mk, action);
                return d;
            } finally {
                RUNNING.remove(key);
            }
        });
    }

    static boolean mayTry(String m, String action, int everyMin, int most) {
        long[] a = ATTEMPTS.get(m + " " + action + " " + today(m));
        if (a == null) return true;
        if (most > 0 && a[0] >= most) return false;
        return a[1] == 0 || System.nanoTime() - a[1] >= everyMin * 60L * 1_000_000_000L;
    }

    static boolean running(String m, String action) {
        return RUNNING.contains(m + " " + action);
    }

    static Path holdState(String m) {
        return Path.of(HISTORY, "holds-" + m + ".json");
    }

    /**
     * Each multi-day book against IBKR's positions: a share the book holds that IBKR does not, one IBKR holds that the
     * book bought but has not recorded (its plan kept), and one IBKR holds that no book has (bought by hand, or lost).
     */
    @SuppressWarnings("unchecked")
    static Object[] inSync(List<Object> positions) {
        Set<String> atIbkr = new LinkedHashSet<>();
        for (Object o : positions) if (o instanceof Map) atIbkr.add(String.valueOf(((Map<String, Object>) o).get("symbol")));
        Set<String> anyBook = new LinkedHashSet<>();
        List<String> wrong = new ArrayList<>();
        for (String m : AUTO) {
            Map<String, Object> st;
            try {
                if (!Files.exists(holdState(m))) continue;
                st = JsonParse.object(Files.readString(holdState(m), StandardCharsets.UTF_8));
            } catch (IOException | RuntimeException e) {
                wrong.add(m + ": its book unreadable (" + e.getMessage() + ")");
                continue;
            }
            Set<String> book = st.get("positions") instanceof Map ? ((Map<String, Object>) st.get("positions")).keySet() : Set.of();
            Set<String> plans = st.get("plans") instanceof Map ? ((Map<String, Object>) st.get("plans")).keySet() : Set.of();
            anyBook.addAll(book);
            List<String> gone = new ArrayList<>(), unrec = new ArrayList<>();
            for (String s : book) if (!atIbkr.contains(s)) gone.add(s);
            for (String s : plans) if (atIbkr.contains(s) && !book.contains(s)) unrec.add(s);
            anyBook.addAll(unrec);
            if (!gone.isEmpty()) wrong.add(m + ": in the book, not at IBKR: " + String.join(" ", gone));
            if (!unrec.isEmpty()) wrong.add(m + ": at IBKR, bought by the book, not recorded yet: " + String.join(" ", unrec));
        }
        List<String> none = new ArrayList<>();
        for (String s : atIbkr) if (!anyBook.contains(s)) none.add(s);
        if (!none.isEmpty()) wrong.add("at IBKR, in no book (the day's buys not yet recorded, or bought by hand): " + String.join(" ", none));
        return wrong.isEmpty() ? new Object[] {true, "every book's positions held at IBKR, and every IBKR position in a book (" + atIbkr.size() + ")"}
                : new Object[] {null, String.join("; ", wrong)};
    }

    @SuppressWarnings("unchecked")
    static Set<String> holdSymbols(String m) {
        try {
            Object p = JsonParse.object(Files.readString(holdState(m), StandardCharsets.UTF_8)).get("positions");
            return p instanceof Map ? new LinkedHashSet<>(((Map<String, Object>) p).keySet()) : Set.of();
        } catch (IOException | RuntimeException e) {
            return Set.of();
        }
    }

    /** Whether the market's live orders are a multi-day book; read once a day (a failed read is tried again later). */
    static boolean liveIsHold(String m) {
        String day = today(m);
        Object[] c = LIVE_HOLD.get(m);
        if (c != null && c[0].equals(day)) return (boolean) c[1];
        try {
            boolean v = "hold".equals(orders(m).get("book"));
            LIVE_HOLD.put(m, new Object[] {day, v});
            return v;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    static boolean holdBook(String m) {
        return liveIsHold(m) || HOLD_BOOKS.containsKey(m);
    }

    static Done done(Steps.Result r) {
        return new Done(r.code() == 0, r.last());
    }

    /** The day's placing: a multi-day book is placed by hold-place; orders from more than 4 days ago are not placed. */
    @SuppressWarnings("unchecked")
    static Done place(String m) {
        try {
            Map<String, Object> book = orders(m);
            String asOf = String.valueOf(book.getOrDefault("asOf", ""));
            if (asOf.isEmpty() || asOf.equals("null") || LocalDate.parse(asOf).isBefore(marketDay(m).minusDays(4)))
                return new Done(false, "orders from " + (asOf.isEmpty() || asOf.equals("null") ? "?" : asOf) + " are too old: not placed");
            if ("hold".equals(book.get("book"))) {
                LIVE_HOLD.put(m, new Object[] {today(m), true});
                markDone(m, "place");
                int n = book.get("orders") instanceof List ? ((List<Object>) book.get("orders")).size() : 0;
                return new Done(true, "a multi-day book (" + n + " orders, held up to " + Holds.g(num(book.getOrDefault("days", 5L), 5)) + " days): placed by hold-place");
            }
            return new Done(false, "not a multi-day book (\"book\": \"hold\"): the bridge places only multi-day books");
        } catch (IOException | RuntimeException e) {
            return new Done(false, e.getMessage());
        }
    }

    static double num(Object o, double fallback) {
        return o instanceof Number ? ((Number) o).doubleValue() : fallback;
    }

    static Steps.Result holds(String m, String action, Map<String, Object> book) throws IOException {
        try (Ib ib = Ib.connect(Recordings.clientId(m) + 7)) {
            Holds h = new Holds(ib, m, action, holdState(m), 1000.0, marketDay(m));
            String last;
            try {
                last = h.run(book);
            } finally {
                Mirror.changed();  // the book's positions and stops: in the store within a minute
            }
            return new Steps.Result(0, new ArrayList<>(h.lines()), last);
        }
    }

    /** The multi-day book's orders for the day: the live orders when they are a hold book, else the shadow's named in HOLD_BOOKS. */
    static Done placeHolds(String m) {
        Steps.Result r = step(() -> {
            Map<String, Object> book;
            if (liveIsHold(m)) {
                book = orders(m);
            } else {
                String name = HOLD_BOOKS.get(m);
                byte[] body = Store.get("shadow/" + m + "-" + name + ".json");
                if (body == null) return new Steps.Result(1, List.of(), "no shadow/" + m + "-" + name + ".json in the store yet");
                Files.createDirectories(Path.of(CACHE));
                Files.write(Path.of(CACHE, m + "-" + name + ".json"), body);
                book = JsonParse.object(new String(body, StandardCharsets.UTF_8));
            }
            if (!marketBuys(m)) {  // no new buys in this market: its book's updates only (stops moved, shares the strategy drops sold)
                book = new LinkedHashMap<>(book);
                book.put("orders", new ArrayList<>());
            }
            return holds(m, "place", book);
        });
        return new Done(r.code() == 0 && !refused(r.out()), r.last());
    }

    /** The buy-back fills since the last run made positions with their own stop at once, every 5 minutes; logged only when it did something. */
    static Done protectHolds(String m) {
        Steps.Result r = step(() -> holds(m, "protect", null));
        List<String> acted = new ArrayList<>();
        for (String x : r.lines()) if (x.contains("bought back") || x.contains("NO STOP") || x.contains("partly filled")) acted.add(x.trim());
        if (r.code() != 0 || !acted.isEmpty()) log(m, "hold-protect", r.code() == 0, acted.isEmpty() ? r.last() : String.join("; ", acted));
        return done(r);
    }

    static void protectSoon(String m) {
        if (!PROTECT_RUNNING.add(m)) return;
        Thread t = new Thread(() -> {
            try {
                protectHolds(m);
            } catch (RuntimeException e) {
                err("hold-protect " + m + ": " + e.getMessage());
            } finally {
                PROTECT_RUNNING.remove(m);
            }
        }, "protect-" + m);
        t.setDaemon(true);
        t.start();
    }

    static Done closeHolds(String m) {
        return done(step(() -> holds(m, "close", null)));
    }

    /** The close's catch-up (its 9-minute window missed): the day's buys recorded and their stops placed; then the close counts as done. */
    static Done recordHolds(String m) {
        Done d = done(step(() -> holds(m, "record", null)));
        if (d.ok()) markDone(m, "hold-close");
        return d;
    }

    /** The shares the model placed in this market lately (the history): the only holdings the leftovers step may sell (never the book's). */
    @SuppressWarnings("unchecked")
    static List<String> modelTickers(String m) {
        String since = LocalDate.now().minusDays(10).toString();
        Set<String> out = new TreeSet<>();
        for (String line : historyLines()) {
            Map<String, Object> r = parse(line);
            if (r == null || !"placed".equals(r.get("kind")) || !m.equals(r.get("market"))) continue;
            if (String.valueOf(r.getOrDefault("date", "")).compareTo(since) < 0) continue;
            Object os = r.get("orders");
            if (os instanceof List) for (Object o : (List<Object>) os) {
                if (o instanceof Map && ((Map<String, Object>) o).get("ticker") != null) out.add(String.valueOf(((Map<String, Object>) o).get("ticker")));
            }
        }
        Set<String> keep = holdSymbols(m);
        List<String> r = new ArrayList<>();
        for (String t : out) if (!keep.contains(t.substring(t.lastIndexOf(':') + 1))) r.add(t);
        return r;
    }

    static Steps.Result leftovers(String m, List<String> tickers, boolean check) {
        return step(() -> {
            try (Ib ib = Ib.connect(Recordings.clientId(m) + 5)) {
                return Steps.leftovers(ib, m, String.join(",", tickers), check, marketDay(m));
            }
        });
    }

    static Done sellLeftovers(String m) {
        List<String> t = modelTickers(m);
        if (t.isEmpty()) return new Done(true, "none placed lately");
        return done(leftovers(m, t, false));
    }

    /** After the close: is any of the model's shares still held? Held is a failure (the monitor turns red); the next morning's leftovers sells it. */
    static Done flatCheck(String m) {
        List<String> t = modelTickers(m);
        if (t.isEmpty()) {
            markDone(m, "flat");
            return new Done(true, "none placed lately");
        }
        Steps.Result r = leftovers(m, t, true);
        if ((r.code() == 0 || r.code() == 1) && r.last().contains("held")) markDone(m, "flat");
        return new Done(r.code() == 0, r.last() + (r.code() == 1 ? ": sold at the next open" : ""));
    }

    /** The day's close: today's positions sold at the close, then the model's shares still held from an earlier day sold too. */
    static Done closeMarket(String m) {
        Steps.Result r = step(() -> {
            try (Ib ib = Ib.connect(Recordings.clientId(m))) {
                return Steps.close(ib, m, marketDay(m));
            }
        });
        String last = r.last();
        int code = r.code();
        List<String> t = modelTickers(m);
        if (!t.isEmpty()) {
            Steps.Result r2 = leftovers(m, t, false);
            String l2 = r2.last();
            int i = l2.indexOf(':');
            if (l2.contains("held from earlier days") && i >= 0 && !l2.substring(i + 1).trim().startsWith("0 held")) last += "; " + l2;
            code = code != 0 ? code : r2.code();
        }
        return new Done(code == 0, last);
    }

    /** Today's fills from IBKR into the history. */
    static Done report(String m) {
        Map<String, Object>[] rec = new Map[1];
        Steps.Result r = step(() -> {
            try (Ib ib = Ib.connect(Recordings.clientId(m))) {
                rec[0] = Steps.report(ib, m, marketDay(m));
                return new Steps.Result(0, List.of(), "");
            }
        });
        if (r.code() != 0 || rec[0] == null) return new Done(false, r.last().isEmpty() ? "no report" : r.last());
        Map<String, Object> out = new LinkedHashMap<>(rec[0]);
        out.put("kind", "result");
        try {
            historyAdd(out);
        } catch (IOException e) {
            return new Done(false, "report not kept: " + e.getMessage());
        }
        return new Done(true, ((List<?>) out.get("trades")).size() + " shares traded");
    }

    // --- IBKR's market data lines: the recordings share a budget (MARKET_LINES), each its share among the overlapping automatic markets
    static final Map<String, List<String>> OVERLAP = Map.of("UK", List.of("UK", "EU", "US", "INDEX"), "EU", List.of("UK", "EU", "US", "INDEX"),
            "US", List.of("UK", "EU", "US", "INDEX"), "INDEX", List.of("UK", "EU", "US", "INDEX"), "APAC", List.of("APAC"));

    static int linesGrant(String name, int want) {
        synchronized (START_LOCK) {
            int used = 0;
            for (Map.Entry<String, Integer> e : LINES_USED.entrySet()) if (!e.getKey().equals(name)) used += e.getValue();
            int n = Math.max(0, Math.min(want, LINES_BUDGET - used));
            LINES_USED.put(name, n);
            return n;
        }
    }

    static void linesRelease(String name) {
        synchronized (START_LOCK) {
            LINES_USED.remove(name);
        }
    }

    static int linesWant(String m, int orders) {
        int together = 0;
        for (String x : OVERLAP.get(m)) if (AUTO.contains(x)) together++;
        return Math.max(LINES_BUDGET / Math.max(1, together), orders);
    }

    /** The candidates' opening auction, kept with the history and in the store as auction/<MARKET>/<date>.json. */
    static Done recordAuction(String m) {
        String name = "auction-" + m;
        Map<String, Object> book;
        try {
            book = orders(m);
        } catch (IOException e) {
            return new Done(false, e.getMessage());
        }
        int n = linesGrant(name, linesWant(m, 0));
        if (n == 0) {
            linesRelease(name);
            return new Done(true, "skipped: no market data lines free (the paths hold them; without real-time data on the account it records nothing anyway)");
        }
        Steps.Result r;
        try {
            r = step(() -> Recordings.auction(m, book, n));
        } finally {
            linesRelease(name);
        }
        if (r.code() != 0) return new Done(false, r.last());
        String line = r.last();
        Map<String, Object> rec = JsonParse.object(line);
        String summary = r.lines().isEmpty() ? ((List<?>) rec.get("shares")).size() + " shares" : r.lines().get(0);
        try {
            Files.createDirectories(Path.of(HISTORY, "auction"));
            Files.writeString(Path.of(HISTORY, "auction", m + "-" + rec.get("date") + ".json"), line + "\n", StandardCharsets.UTF_8);
            Store.put("auction/" + m + "/" + rec.get("date") + ".json", (line + "\n").getBytes(StandardCharsets.UTF_8), m + " opening auction " + rec.get("date"));
        } catch (IOException e) {
            return new Done(true, summary + "; kept here, not stored in svr-net/eod (" + e.getMessage() + ")");
        }
        return new Done(true, summary);
    }

    /** Today's candidates' prices through the day, written to the store as paths/<MARKET>/<date>.csv. Runs the whole session. */
    @SuppressWarnings("unchecked")
    static Done recordPaths(String m) {
        if (!PATHS_RUNNING.add(m)) return new Done(false, "already running");
        String name = "paths-" + m;
        try {
            String day = today(m);
            Files.createDirectories(Path.of(HISTORY));
            Path out = Path.of(HISTORY, "paths-" + m + "-" + day + ".csv");
            Map<String, Object> book = orders(m);
            int n = linesGrant(name, linesWant(m, ((List<Object>) book.getOrDefault("orders", List.of())).size()));
            if (n == 0) return new Done(false, "no market data lines free");
            Steps.Result r = Recordings.paths(m, book, n, out);  // an all-day watch: the bars fetch is not paused for it
            if (r.code() != 0 || !Files.exists(out)) return new Done(false, r.last().isEmpty() ? "failed" : r.last());
            Store.put("paths/" + m + "/" + day + ".csv", Files.readAllBytes(out), m + " intraday paths " + day);
            return new Done(true, r.last());
        } catch (IOException | RuntimeException e) {
            return new Done(false, e.getMessage());
        } finally {
            linesRelease(name);
            PATHS_RUNNING.remove(m);
        }
    }

    /**
     * IBKR's daily bars for the pool, merged into the file kept here and written to the store as ibkr/<MARKET>/daily.csv. daily: only the
     * latest month. Every fetch is merged in (a share-day IBKR answers again replaces the kept one), so a fetch IBKR answered badly never
     * loses bars (7 Nov); without a kept file the store's is the start; a fetch with no bars changes nothing.
     */
    static Done fetchBars(String m, boolean daily) {
        synchronized (START_LOCK) {
            if (!BARS_RUNNING.isEmpty()) return new Done(false, "waiting: " + String.join(", ", new TreeSet<>(BARS_RUNNING)) + " bars fetching");
            BARS_RUNNING.add(m);
        }
        try {
            Path outFile = Path.of(HISTORY, "ibkr-" + m + ".csv");
            Files.createDirectories(Path.of(HISTORY));
            if (!(Files.exists(outFile) && Files.size(outFile) > 100)) {
                byte[] kept = Store.get("ibkr/" + m + "/daily.csv");
                if (kept != null && kept.length > 100) Files.write(outFile, kept);
            }
            boolean have = Files.exists(outFile) && Files.size(outFile) > 100;
            Path target = Path.of(outFile + ".new");
            PAUSE.clearIfFree();
            Steps.Result r = Recordings.bars(m, orders(m), daily && have ? "1 M" : "2 Y", target, PAUSE, Auto::err);
            String summary = r.last();
            if (r.code() != 0) return new Done(false, summary.isEmpty() ? "failed" : summary);
            List<String[]> fresh = Csv.read(target);
            if (fresh.size() < 2) return new Done(false, summary + ": no bars from IBKR; the kept file unchanged");
            List<String[]> rows = new ArrayList<>();
            if (have) {
                Set<String> keys = new java.util.HashSet<>();
                for (String[] x : fresh.subList(1, fresh.size())) if (x.length >= 2) keys.add(day(x[0]) + "\u0000" + x[1]);
                List<String[]> old = Csv.read(outFile);
                for (String[] x : old.subList(Math.min(1, old.size()), old.size())) if (x.length >= 2 && !keys.contains(day(x[0]) + "\u0000" + x[1])) rows.add(x);
            }
            rows.addAll(fresh.subList(1, fresh.size()));
            rows.sort((a, b) -> {
                int c = day(a[0]).compareTo(day(b[0]));
                return c != 0 ? c : a[1].compareTo(b[1]);
            });
            Path tmp = Path.of(outFile + ".tmp");
            List<String[]> all = new ArrayList<>();
            all.add(fresh.get(0));
            all.addAll(rows);
            Csv.write(tmp, all);
            Files.move(tmp, outFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            Set<String> shares = new java.util.HashSet<>();
            for (String[] x : rows) shares.add(x[1]);
            summary += "; " + rows.size() + " rows kept, " + shares.size() + " shares, the latest " + day(rows.get(rows.size() - 1)[0]);
            Store.put("ibkr/" + m + "/daily.csv", Files.readAllBytes(outFile), m + " daily bars from IBKR " + LocalDate.now());
            return new Done(true, summary);
        } catch (IOException | RuntimeException e) {
            return new Done(false, e.getMessage());
        } finally {
            BARS_RUNNING.remove(m);
        }
    }

    static String day(String d) {
        return d.length() > 10 ? d.substring(0, 10) : d;
    }

    // --- the background reads: requests answer at once from the latest snapshot
    static void readOnce(String name, Runnable fn) {
        if (!READING.add(name)) return;
        Thread t = new Thread(() -> {
            try {
                fn.run();
            } catch (RuntimeException e) {
                err("background read " + name + ": " + e.getMessage());
            } finally {
                READING.remove(name);
            }
        }, "read-" + name);
        t.setDaemon(true);
        t.start();
    }

    /** Today's trades and the model's holdings from IBKR, into TRADES. */
    static void readTrades(String m) {
        Set<String> tickers = new TreeSet<>(modelTickers(m));
        tickers.addAll(holdSymbols(m));
        Map<String, Object>[] got = new Map[1];
        Steps.Result r = step(() -> {
            try (Ib ib = Ib.connect(Bridge.env("IB_HOST", "127.0.0.1"), IB_PORT, Recordings.clientId(m) + 6, 2)) {
                got[0] = Steps.state(ib, m, String.join(",", tickers), CAPITAL, marketDay(m));
                return new Steps.Result(0, List.of(), "");
            }
        });
        if (r.code() != 0 || got[0] == null) {
            TRADES_NOTE.put(m, r.last().isEmpty() ? "no answer from IBKR" : r.last());
            return;
        }
        TRADES.put(m, new Object[] {System.nanoTime(), got[0]});
        TRADES_NOTE.remove(m);
    }

    static void readIbdiag() {
        String m = AUTO.isEmpty() ? "UK" : AUTO.get(0);
        Map<String, Object> d;
        PAUSE.hold();
        try {
            d = Recordings.diag(m);
        } catch (RuntimeException e) {
            d = new LinkedHashMap<>();
            d.put("api", false);
            d.put("error", e.getMessage());
        } finally {
            PAUSE.release();
        }
        ibdiag = d;
        ibdiagAt = System.nanoTime();
    }

    /** The IBKR check is due: every 5 minutes, every minute while it last failed (IB Gateway still logging in after a start). */
    static boolean diagDue() {
        Map<String, Object> d = ibdiag;
        return ibdiagAt == null || ago(ibdiagAt) >= (d != null && Boolean.TRUE.equals(d.get("api")) ? 300 : 60);
    }

    static long ago(long nanos) {
        return (System.nanoTime() - nanos) / 1_000_000_000L;
    }

    static void refresher() {
        while (true) {
            for (String m : AUTO) {
                try {
                    ZonedDateTime now = now(m);
                    String[] t = TIMES.get(m);
                    if (now.getDayOfWeek().getValue() < 6 && between(now, at(now, t[0], -30), at(now, t[1], 30))) {
                        Object[] hit = TRADES.get(m);
                        if (hit == null || ago((long) hit[0]) >= 55) readOnce("trades-" + m, () -> readTrades(m));
                    }
                } catch (RuntimeException e) {
                    err("refresher " + m + ": " + e.getMessage());
                }
            }
            if (diagDue()) readOnce("diag", Auto::readIbdiag);
            beat("refresher");
            Recordings.sleep(20);
        }
    }

    /** Every minute, whether IB Gateway's API answers and still reaches IBKR's servers; 3 minutes in a row with its port open but its API silent: restarted. */
    static void watchdog() {
        while (true) {
            String why;
            try {
                why = Recordings.ping();
            } catch (RuntimeException e) {
                why = "no answer: " + e.getMessage();
            }
            boolean ok = why.isEmpty();
            boolean hung = !ok && gatewayUp();  // a closed port is the gateway starting or restarting
            String checked = stamp(ZoneId.systemDefault().getId());
            GATEWAY.put("ok", ok);
            GATEWAY.put("checked", checked);
            GATEWAY.put("fails", hung ? (int) GATEWAY.get("fails") + 1 : 0);
            // a closed port is the gateway starting, but not for 10 minutes: 24-25 Nov its port stayed closed from 22:38
            // to 08:04 GMT after a restart (the login never finished) and nothing restarted it
            int closed = !ok && !hung ? (int) GATEWAY.getOrDefault("closed", 0) + 1 : 0;
            GATEWAY.put("closed", closed);
            // a bars fetch slows the gateway's answers: 6 misses in a row then, not 3, before a restart
            int most = BARS_RUNNING.isEmpty() ? 3 : 6;
            if ((int) GATEWAY.get("fails") >= most || closed >= 10) {
                String what = closed >= 10 ? "IB Gateway's port closed for " + closed + " minutes" : "IB Gateway did not answer " + most + " times";
                GATEWAY.put("closed", 0);
                String note = restartGateway();
                GATEWAY.put("fails", 0);
                GATEWAY.put("restarts", (int) GATEWAY.get("restarts") + 1);
                GATEWAY.put("restarted", checked);
                GATEWAY.put("note", note);
                synchronized (LAST) {
                    LAST.add(0, new Action(checked, "IB", "gateway restart", note.contains("restarted") || note.contains("stopped"), what + ": " + note));
                    while (LAST.size() > 10) LAST.remove(LAST.size() - 1);
                }
                err(what + ": " + note);
                beat("watchdog");
                Recordings.sleep(240);
                continue;
            }
            beat("watchdog");
            Recordings.sleep(60);
        }
    }

    /** The repository's switch for the automatic buying (config/nas-buys.json), read every 5 minutes; a failed read keeps the last value. */
    static volatile Set<String> buysOff;

    /** Whether the market may buy (config/nas-buys.json's "markets": {"EU": false}): off, its book still updated (stops, exits). */
    static boolean marketBuys(String m) {
        buysOn();
        if (buysOff == null) {
            try {
                String t = Files.readString(Path.of(HISTORY, "buys-off-markets")).strip();
                buysOff = t.isEmpty() ? Set.of() : new TreeSet<>(List.of(t.split(",")));
            } catch (IOException e) {
                buysOff = Set.of();
            }
        }
        return !buysOff.contains(m);
    }

    static boolean buysOn() {
        Path keep = Path.of(HISTORY, "buys-on");
        if (buysOn == null) {
            try {
                buysOn = !Files.readString(keep).strip().equals("0");
            } catch (IOException e) {
                buysOn = true;
            }
        }
        if (System.nanoTime() - buysChecked >= 300L * 1_000_000_000L) {
            buysChecked = System.nanoTime();
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(BUYS_URL).openConnection();
                c.setConnectTimeout(20_000);
                c.setReadTimeout(20_000);
                c.setRequestProperty("User-Agent", "ofm-bridge");
                c.setRequestProperty("Cache-Control", "no-cache");
                if (c.getResponseCode() != 200) throw new IOException("HTTP Error " + c.getResponseCode());
                boolean on;
                Set<String> off = new TreeSet<>();
                try (InputStream in = c.getInputStream()) {
                    Map<String, Object> j = JsonParse.object(new String(in.readAllBytes(), StandardCharsets.UTF_8));
                    on = !Boolean.FALSE.equals(j.get("buys"));
                    if (j.get("markets") instanceof Map)
                        for (Map.Entry<?, ?> e : ((Map<?, ?>) j.get("markets")).entrySet()) if (Boolean.FALSE.equals(e.getValue())) off.add(String.valueOf(e.getKey()));
                }
                buysOff = off;
                Files.createDirectories(Path.of(HISTORY));
                Files.writeString(Path.of(HISTORY, "buys-off-markets"), String.join(",", off));
                buysOn = on;
                buysNote = (on ? "on" : "off") + (off.isEmpty() ? "" : ", no new buys in " + String.join(", ", off)) + " (config/nas-buys.json, read " + ZonedDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm z")) + ")";
                Files.createDirectories(Path.of(HISTORY));
                Files.writeString(keep, on ? "1" : "0");
                Mirror.changed();
            } catch (IOException | RuntimeException e) {
                buysNote = (buysOn ? "on" : "off") + " (the last read; config/nas-buys.json not read: " + e.getClass().getSimpleName() + ")";
            }
        }
        return buysOn;
    }

    static boolean stepsPending() {
        for (String k : RUNNING) {
            String a = k.substring(k.indexOf(' ') + 1);
            if (List.of("leftovers", "place", "close", "flat", "report", "hold-place", "hold-close", "hold-record").contains(a)) return true;
        }
        for (String m : AUTO) {
            ZonedDateTime now = now(m);
            if (now.getDayOfWeek().getValue() < 6 && within(now, REPORT.get(m), 0, 180) && !isDone(m, "report")) return true;
        }
        return false;
    }

    static void scheduler() {
        try {
            Files.createDirectories(Path.of(CACHE));
        } catch (IOException e) {
            err("cache: " + e.getMessage());
        }
        while (true) {
            for (String m : AUTO) {
                try {
                    tick(m);
                } catch (RuntimeException e) {
                    err("scheduler " + m + ": " + e);
                }
            }
            beat("scheduler");
            Recordings.sleep(30);
        }
    }

    static void tick(String m) {
        ZonedDateTime now = now(m);
        String day = today(m);
        if (now.getDayOfWeek() == DayOfWeek.SATURDAY && !BARS_AT.isEmpty() && within(now, BARS_AT, 0, 240) && !isDone(m, "bars") && BARS_RUNNING.isEmpty() && mayTry(m, "bars", 10, 0))
            job(m, "bars", mk -> fetchBars(mk, false), true);
        if (now.getDayOfWeek().getValue() >= 6 && System.getenv("AUTO_WEEKENDS") == null) return;
        String place = TIMES.get(m)[0], close = TIMES.get(m)[1];
        boolean up = !Boolean.FALSE.equals(GATEWAY.get("ok"));
        if (BARS_DAILY && up && within(now, REPORT.get(m), 15, 360) && !isDone(m, "bars") && BARS_RUNNING.isEmpty() && !stepsPending() && mayTry(m, "bars", 20, 4))
            job(m, "bars", mk -> {
                // IBKR refusing the data (162: another session holds it) is not cured by asking again in 20 minutes: the
                // day's daily fetch stops there (25 Nov: 4 refused fetches a market an evening, and the gateway hung twice)
                Done d = fetchBars(mk, true);
                if (!d.ok() && d.summary().contains("162:")) markDone(mk, "bars");
                return d;
            }, true);
        boolean session = between(now, at(now, place, -30), at(now, close, -15));
        if (up && session && !isDone(m, "leftovers") && mayTry(m, "leftovers", 5, 3)) job(m, "leftovers", Auto::sellLeftovers, true);
        long[] lt = ATTEMPTS.get(m + " leftovers " + day);
        boolean settled = isDone(m, "leftovers") || (lt != null && lt[0] > 0 && !running(m, "leftovers"));
        boolean placing = between(now, at(now, place, 0), at(now, close, -15));
        if (up && settled && placing && !isDone(m, "place") && buysOn() && mayTry(m, "place", 5, 4)) job(m, "place", Auto::place, false);
        String start = AUCTION.getOrDefault(m, place);
        if (PATHS && up && between(now, at(now, start, 0), at(now, close, 15)) && !PATHS_RUNNING.contains(m) && !isDone(m, "paths") && mayTry(m, "paths", 5, 0))
            job(m, "paths", Auto::recordPaths, true);
        if (AUCTION.containsKey(m) && up && within(now, AUCTION.get(m), 0, 5) && !isDone(m, "auction") && mayTry(m, "auction", 10, 1))
            job(m, "auction", Auto::recordAuction, true);
        if (within(now, close, 0, 9) && !isDone(m, "close") && mayTry(m, "close", 1, 0)) job(m, "close", Auto::closeMarket, true);
        boolean hold = holdBook(m);
        if (hold && up && isDone(m, "place") && placing && !isDone(m, "hold-place") && !running(m, "place") && !PROTECT_RUNNING.contains(m) && buysOn() && mayTry(m, "hold-place", 5, 3))
            job(m, "hold-place", Auto::placeHolds, true);
        if (hold && up && within(now, close, 0, 9) && !isDone(m, "hold-close") && !PROTECT_RUNNING.contains(m) && mayTry(m, "hold-close", 2, 3))
            job(m, "hold-close", Auto::closeHolds, true);
        // its window missed (IB Gateway down or hung at the close, 24 Nov): caught up within 6 hours of the close
        if (hold && up && between(now, at(now, close, 10), at(now, close, 360)) && isDone(m, "hold-place") && !isDone(m, "hold-close")
                && !running(m, "hold-close") && !PROTECT_RUNNING.contains(m) && mayTry(m, "hold-record", 10, 3))
            job(m, "hold-record", Auto::recordHolds, false);
        if (hold && up && between(now, at(now, place, 0), at(now, close, 0)) && !running(m, "hold-place") && !running(m, "hold-close") && mayTry(m, "hold-protect", 5, 0)) {
            ATTEMPTS.put(m + " hold-protect " + day, new long[] {0, System.nanoTime()});
            protectSoon(m);
        }
        if (up && within(now, close, 30, 240) && !isDone(m, "flat") && mayTry(m, "flat", 10, 0)) job(m, "flat", Auto::flatCheck, false);
        if (within(now, REPORT.get(m), 0, 180) && !isDone(m, "report") && mayTry(m, "report", 10, 0)) job(m, "report", Auto::report, true);
    }

    static volatile String restored = "not started";

    static void start() {
        restored = Mirror.restore();  // before anything reads the history
        err("state: " + restored);
        Mirror.start(restored);
        if (!AUTO.isEmpty()) {
            seedDone();
            daemon("scheduler", Auto::scheduler);
            daemon("watchdog", Auto::watchdog);
        }
        daemon("refresher", Auto::refresher);
    }

    /**
     * A loop that must never stop (the scheduler, the watchdog, the background reads): anything it throws is logged and the loop starts
     * again 10 seconds later. Out of memory ends the bridge (-XX:+ExitOnOutOfMemoryError), which ibkr-start.sh starts again.
     */
    static void daemon(String name, Runnable r) {
        Thread t = new Thread(() -> {
            while (true) {
                try {
                    r.run();
                    err(name + " ended; started again in 10 seconds");
                } catch (Throwable e) {  // NOPMD: the loop outlives anything but the JVM
                    err(name + " failed (" + e + "); started again in 10 seconds");
                }
                Recordings.sleep(10);
            }
        }, name);
        t.setDaemon(true);
        t.start();
    }

    // --- liveness: each loop's last turn; GET /live answers 503 when one is late (ibkr-start.sh then restarts the bridge)
    static final Map<String, Long> BEATS = new ConcurrentHashMap<>();
    static final Map<String, Integer> LATE_AFTER = Map.of("scheduler", 300, "watchdog", 600, "refresher", 300);

    static void beat(String loop) {
        BEATS.put(loop, System.nanoTime());
    }

    /** The loops late by more than their limit (seconds since their last turn), or none: the bridge is alive. */
    static Map<String, Long> late() {
        Map<String, Long> out = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : LATE_AFTER.entrySet()) {
            if (AUTO.isEmpty() && !e.getKey().equals("refresher")) continue;  // no automatic markets: no scheduler or watchdog
            Long b = BEATS.get(e.getKey());
            long age = b == null ? ago(STARTED_NANOS) : ago(b);
            if (age > e.getValue()) out.put(e.getKey(), age);
        }
        return out;
    }

    static final long STARTED_NANOS = System.nanoTime();

    // --- GET /status (no key: tells the page the bridge is there) and GET /diag
    static final List<String> TODAY = List.of("leftovers", "place", "paths", "auction", "close", "flat", "report");

    static List<String> runningOf(String m) {
        List<String> r = new ArrayList<>();
        for (String k : RUNNING) if (k.startsWith(m + " ")) r.add(k.substring(m.length() + 1));
        Collections.sort(r);
        return r;
    }

    static Map<String, Object> status() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("bridge", true);
        out.put("version", Bridge.VERSION);
        out.put("started", STARTED.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z")));
        out.put("gateway", gatewayUp());
        out.put("port", IB_PORT);
        out.put("capital", CAPITAL);
        out.put("keyed", !Bridge.KEY.isEmpty());
        out.put("java", true);
        Map<String, Object> auto = new LinkedHashMap<>();
        for (String m : AUTO) {
            Map<String, Object> a = new LinkedHashMap<>();
            a.put("place", TIMES.get(m)[0]);
            a.put("auction", AUCTION.get(m));
            a.put("close", TIMES.get(m)[1]);
            a.put("report", REPORT.get(m));
            a.put("zone", ZONE.get(m));
            Map<String, Object> t = new LinkedHashMap<>();
            for (String s : TODAY) t.put(s, isDone(m, s));
            a.put("today", t);
            a.put("running", runningOf(m));
            auto.put(m, a);
        }
        out.put("auto", auto);
        synchronized (GATEWAY) {
            out.put("gatewayApi", new LinkedHashMap<>(GATEWAY));
        }
        List<Object> last = new ArrayList<>();
        for (Action x : last()) {
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("when", x.when());
            e.put("market", x.market());
            e.put("action", x.action());
            e.put("ok", x.ok());
            e.put("summary", x.summary());
            last.add(e);
        }
        out.put("last", last);
        return out;
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> diag() {
        List<Map<String, Object>> out = new ArrayList<>();
        java.util.function.BiConsumer<String, Object[]> add = (check, v) -> {
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("check", check);
            e.put("ok", v[0]);
            e.put("detail", v[1]);
            out.add(e);
        };
        add.accept("Bridge", new Object[] {true, "version " + Bridge.VERSION + " (Java, IBKR API " + Api.version() + "), running since "
                + STARTED.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z")) + "; automatic: " + (AUTO.isEmpty() ? "off" : String.join(",", AUTO)) + "; capital " + Holds.g(CAPITAL) + " a trade"});
        Map<String, Long> late = late();
        add.accept("Bridge loops", new Object[] {late.isEmpty(), late.isEmpty() ? "the scheduler, the watchdog and the background reads turning"
                : "late (seconds since their last turn): " + late + "; the container restarts the bridge after 3 minutes of this"});
        add.accept("State in the store", new Object[] {!Mirror.ON ? null : Mirror.held ? Boolean.FALSE : Mirror.ok, !Mirror.ON ? "off (STATE_MIRROR=0)"
                : "svr-net/eod " + Mirror.PATH + ": " + Mirror.note + (Mirror.dirty ? "; changes waiting" : "") + "; at start: " + restored});
        add.accept("Store cache", new Object[] {Store.lastStale.isEmpty() ? true : null, Store.lastStale.isEmpty()
                ? "every store read from GitHub or its unchanged copy here" : "last served from the copy here: " + Store.lastStale});
        add.accept("Bridge update", new Object[] {true, "newest image checked when no market is near its day: " + UPDATE_NOTE});
        if (Store.TOKEN.isEmpty()) {
            add.accept("EOD_TOKEN", new Object[] {false, "not set in .env"});
        } else {
            try {
                boolean push = Boolean.TRUE.equals(diagMemo("canPush", () -> Store.canPush()));
                add.accept("EOD_TOKEN", new Object[] {push ? true : null, "reads svr-net/eod" + (push ? "" : "; may not write (recordings stay on the NAS)")});
            } catch (Store.Http e) {
                add.accept("EOD_TOKEN", new Object[] {false, e.code == 401 ? "rejected (expired or revoked)" : e.code == 404 ? "no access to svr-net/eod" : "HTTP " + e.code});
            } catch (IOException e) {
                add.accept("EOD_TOKEN", new Object[] {false, "GitHub not reachable (" + e.getMessage() + ")"});
            }
        }
        for (String m : AUTO.isEmpty() ? List.of("UK") : AUTO) {
            try {
                Map<String, Object> book = diagMemo("orders-" + m, () -> orders(m));
                String asOf = String.valueOf(book.getOrDefault("asOf", "1970-01-01"));
                long age = java.time.temporal.ChronoUnit.DAYS.between(LocalDate.parse(asOf), LocalDate.now());
                add.accept(m + " orders", new Object[] {age <= 4 ? true : null, "after the close of " + asOf + " (" + age + " day" + (age != 1 ? "s" : "") + " ago): "
                        + size(book.get("orders")) + " orders, " + size(book.get("candidates")) + " candidates, pool " + size(book.get("pool"))});
            } catch (IOException | RuntimeException e) {
                add.accept(m + " orders", new Object[] {false, e.getMessage()});
            }
        }
        add.accept("IB Gateway port", new Object[] {gatewayUp(), IB_HOST + ":" + IB_PORT});
        String m0 = AUTO.isEmpty() ? "UK" : AUTO.get(0);
        if (diagDue()) readOnce("diag", Auto::readIbdiag);
        Map<String, Object> d = ibdiag;
        String agoNote = ibdiagAt != null ? " (checked " + ago(ibdiagAt) + " s ago)" : "";
        if (d == null) {
            add.accept("IB Gateway API", new Object[] {null, "being checked in the background: ask again in a minute"});
        } else if (!Boolean.TRUE.equals(d.get("api"))) {
            add.accept("IB Gateway API", new Object[] {false, d.getOrDefault("error", "no answer") + agoNote});
        } else {
            add.accept("IB Gateway API", new Object[] {true, "logged in, " + (Boolean.TRUE.equals(d.get("paper")) ? "paper" : "LIVE") + " account; IBKR time "
                    + (d.get("serverTime") == null ? "?" : d.get("serverTime")) + agoNote});
            List<String> errs = (List<String>) d.getOrDefault("errors", List.of());
            boolean live = Integer.valueOf(1).equals(d.get("dataType")) && errs.stream().noneMatch(e -> e.startsWith("354") || e.startsWith("10089") || e.startsWith("10168"));
            List<String> q = new ArrayList<>();
            for (String k : List.of("bid", "ask", "last", "close")) if (d.get(k) != null) q.add(k + " " + Holds.g(((Number) d.get(k)).doubleValue()));
            add.accept(m0 + " real-time data", new Object[] {live ? true : null, d.getOrDefault("probe", "?") + ": " + (q.isEmpty() ? "no prices (outside hours, or no subscription)" : String.join(", ", q))
                    + (live ? "" : "; the auction recording needs the exchange's real-time subscription")});
            int hb = d.get("historyBars") instanceof Number ? ((Number) d.get("historyBars")).intValue() : 0;
            add.accept(m0 + " historical data", new Object[] {hb > 0, hb + " daily bars" + (d.get("historyError") != null ? " (" + d.get("historyError") + ")" : "")});
            if (!errs.isEmpty()) add.accept("IBKR messages", new Object[] {null, String.join(" | ", errs)});
            if (d.get("positions") instanceof List) add.accept("Books against IBKR", inSync((List<Object>) d.get("positions")));
            else if (d.get("positionsError") != null) add.accept("Books against IBKR", new Object[] {null, "not checked: IBKR's positions did not come (" + d.get("positionsError") + "); next check in 5 minutes"});
        }
        List<String> plan = new ArrayList<>();
        for (String m : AUTO) plan.add(m + ": place " + TIMES.get(m)[0] + (AUCTION.containsKey(m) ? ", auction " + AUCTION.get(m) : "") + ", hold book" + (PATHS ? ", paths" : "")
                + ", close " + TIMES.get(m)[1] + ", report " + REPORT.get(m) + " (" + ZONE.get(m) + ")");
        add.accept("Automatic buying", new Object[] {buysOn() ? true : null, buysNote});
        add.accept("Schedule", new Object[] {AUTO.isEmpty() ? null : true, (plan.isEmpty() ? "automatic trading off (AUTO_MARKETS)" : String.join("; ", plan))
                + (BARS_AT.isEmpty() ? "" : "; IBKR bars Saturdays " + BARS_AT)});
        for (String m : AUTO) {
            List<String> steps = new ArrayList<>(TODAY);
            if (holdBook(m)) steps.addAll(List.of("hold-place", "hold-close"));
            List<String> parts = new ArrayList<>();
            for (String a : steps) parts.add(a + " " + (isDone(m, a) ? "done" : "-"));
            List<String> run = runningOf(m);
            if (PROTECT_RUNNING.contains(m)) run.add("hold-protect");
            add.accept(m + " today", new Object[] {null, String.join(", ", parts) + "; running: " + (run.isEmpty() ? "nothing" : String.join(", ", run))});
        }
        Object ok = GATEWAY.get("ok");
        add.accept("IB Gateway watchdog", new Object[] {ok, (ok == null ? "not checked yet" : (boolean) ok ? "API answers" : "API does not answer") + " (checked " + GATEWAY.get("checked")
                + "); restarts " + GATEWAY.get("restarts") + (GATEWAY.get("restarted") != null ? ", last " + GATEWAY.get("restarted") + ": " + GATEWAY.get("note") : "")});
        List<Action> last = last();
        List<Action> answers = new ArrayList<>();
        for (Action x : last) if (x.summary().contains("Real paper-trading has not been acknowledged") || x.summary().contains("2139:") || (x.action().equals("test-order") && x.ok())) answers.add(x);
        boolean refusedNow = !answers.isEmpty() && !(answers.get(0).action().equals("test-order") && answers.get(0).ok());
        if (!answers.isEmpty() && !refusedNow) add.accept("IBKR accepts orders", new Object[] {true, answers.get(0).when() + " " + answers.get(0).market() + ": " + answers.get(0).summary()});
        if (refusedNow) {
            Action a = answers.get(0);
            add.accept("IBKR accepts orders", new Object[] {false, a.when() + " " + a.market() + " " + a.action() + ": IBKR refused the orders (2139: real paper-trading has not been acknowledged): log in to the "
                    + "paper account once (Client Portal or TWS, paper mode) and accept the paper-trading notice, then restart IB Gateway"});
        }
        boolean failed = last.stream().anyMatch(x -> !x.ok());
        List<String> ls = new ArrayList<>();
        for (Action x : last.subList(0, Math.min(8, last.size()))) ls.add(x.when() + " " + x.market() + " " + x.action() + ": " + (x.ok() ? "ok" : "FAILED") + " " + x.summary());
        add.accept("Last automatic actions", new Object[] {failed ? null : true, ls.isEmpty() ? "none yet" : String.join("; ", ls)});
        try {
            Path adir = Path.of(HISTORY, "auction");
            List<String> files = new ArrayList<>();
            if (Files.isDirectory(adir)) try (var s = Files.list(adir)) {
                s.forEach(p -> files.add(p.getFileName().toString()));
            }
            Collections.sort(files);
            if (files.isEmpty()) {
                add.accept("Auction recordings", new Object[] {null, "none yet"});
            } else {
                String f = files.get(files.size() - 1);
                Map<String, Object> rec = JsonParse.object(Files.readString(adir.resolve(f), StandardCharsets.UTF_8));
                List<Object> shares = (List<Object>) rec.get("shares");
                int n = 0;
                for (Object r : shares) {
                    for (Object s : (List<Object>) ((Map<String, Object>) r).get("samples")) {
                        Object pr = ((Map<String, Object>) s).get("price");
                        if (pr instanceof Number && ((Number) pr).doubleValue() != 0) {
                            n++;
                            break;
                        }
                    }
                }
                add.accept("Auction recordings", new Object[] {n > 0 ? true : null, files.size() + " day" + (files.size() != 1 ? "s" : "") + ", latest " + f.substring(0, f.length() - 5)
                        + ": an indicative price for " + n + " of " + shares.size() + " shares"});
            }
        } catch (IOException | RuntimeException e) {
            add.accept("Auction recordings", new Object[] {null, "not read: " + e.getMessage()});
        }
        for (String m : AUTO.isEmpty() ? List.of("UK") : AUTO) {
            Path f = Path.of(HISTORY, "ibkr-" + m + ".csv");
            if (BARS_RUNNING.contains(m)) {
                add.accept(m + " IBKR bars", new Object[] {null, "fetching now"});
            } else if (Files.exists(f)) {
                long rows;
                try (BufferedReader r = Files.newBufferedReader(f, StandardCharsets.UTF_8)) {
                    rows = r.lines().count() - 1;
                } catch (IOException e) {
                    rows = 0;
                }
                String when;
                try {
                    when = ZonedDateTime.ofInstant(Files.getLastModifiedTime(f).toInstant(), ZoneId.systemDefault()).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"));
                } catch (IOException e) {
                    when = "?";
                }
                add.accept(m + " IBKR bars", new Object[] {rows > 0 ? true : null, rows + " bars, fetched " + when});
            } else {
                add.accept(m + " IBKR bars", new Object[] {null, "not fetched yet"});
            }
        }
        try {
            List<Map<String, Object>> days = Book.history();
            add.accept("History", new Object[] {true, days.size() + " market-days recorded" + (days.isEmpty() ? "" : ", latest " + days.get(0).get("date") + " " + days.get(0).get("market"))});
        } catch (IOException e) {
            add.accept("History", new Object[] {null, e.getMessage()});
        }
        return out;
    }

    static int size(Object o) {
        return o instanceof List ? ((List<?>) o).size() : 0;
    }

    private Auto() {
    }
}
