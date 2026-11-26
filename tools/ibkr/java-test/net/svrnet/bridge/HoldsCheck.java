package net.svrnet.bridge;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The hold book (Holds) through tools/ibkr/test/levels.py's hold-book scenarios, against the same stand-in IBKR
 * (FakeBroker), with the same checks: java -cp ... net.svrnet.bridge.HoldsCheck
 */
public final class HoldsCheck {
    static final LocalDate TODAY = LocalDate.now(ZoneOffset.UTC);
    static Path dir;

    static void at(String zone, int h, int m) {
        ZonedDateTime t = ZonedDateTime.of(TODAY, LocalTime.of(h, m), ZoneId.of(zone));
        Holds.clock = Clock.fixed(t.toInstant(), ZoneOffset.UTC);
    }

    static void check(boolean ok, Object why) {
        if (!ok) throw new AssertionError(String.valueOf(why));
    }

    static Map<String, Object> order(String ticker, double close, double limit, Object exitAbove, int days, double capital, double maxCapital) {
        Map<String, Object> o = new LinkedHashMap<>();
        o.put("ticker", ticker);
        o.put("close", close);
        o.put("entry", "open");
        o.put("open", true);
        o.put("limit", limit);
        o.put("stop", null);
        o.put("take", null);
        o.put("exitAbove", exitAbove);
        o.put("days", (long) days);
        o.put("capital", capital);
        o.put("maxCapital", maxCapital);
        return o;
    }

    static Map<String, Object> book(String model, int days, int slots, List<Map<String, Object>> orders) {
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("model", model);
        b.put("book", "hold");
        b.put("days", (long) days);
        b.put("slots", (long) slots);
        b.put("asOf", TODAY.toString());
        b.put("orders", new ArrayList<Object>(orders));
        return b;
    }

    static Map<String, Object> upd(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    static String run(FakeBroker b, String market, String action, Path state, Map<String, Object> book) throws Exception {
        return run(b, market, action, state, book, TODAY);
    }

    static String run(FakeBroker b, String market, String action, Path state, Map<String, Object> book, LocalDate day) throws Exception {
        Holds h = new Holds(b, market, action, state, 5000, day);
        String last = h.run(book == null ? null : JsonParse.object(Json.write(book)));  // as read from the orders file
        StringBuilder s = new StringBuilder();
        for (String l : h.lines()) s.append("  ").append(l).append('\n');
        return s.append(last).append('\n').toString();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> state(Path p) throws Exception {
        return JsonParse.object(Files.readString(p, StandardCharsets.UTF_8));
    }

    static void save(Path p, Map<String, Object> st) throws Exception {
        Files.writeString(p, Json.write(st), StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> pos(Map<String, Object> st, String sym) {
        return (Map<String, Object>) ((Map<String, Object>) st.get("positions")).get(sym);
    }

    static ZonedDateTime until(String gtd, String zone) {
        return java.time.LocalDateTime.parse(gtd, DateTimeFormatter.ofPattern("yyyyMMdd-HH:mm:ss")).atZone(ZoneOffset.UTC).withZoneSameInstant(ZoneId.of(zone));
    }

    // --- levels.py's scenarios ------------------------------------------------------------------------------------
    static String holdBook() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> book = book("uk", 5, 5, List.of(order("XXX", 100, 101, 0.012, 5, 1000, 1000)));
        Path state = dir.resolve("holds-UK.json");
        String placed = run(b, "UK", "place", state, book);
        FakeBroker.T buy = b.find("XXX", "BUY");
        check(buy.order.type().equals("LMT") && buy.order.tif().equals("GTD") && buy.order.limit() == 101 && buy.order.qty() == 990, placed);
        String again = run(b, "UK", "place", state, book);
        check(b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("BUY")).size() == 1 && again.contains("already working at IBKR, kept"), again);
        check(((Map<String, Object>) state(state).get("pending")).containsKey("XXX"), again);
        b.fill(buy, 990, 100.0);
        at("Europe/London", 16, 30);
        String closed = run(b, "UK", "close", state, null);
        FakeBroker.T sell = b.find("XXX", "SELL");
        check(sell != null && sell.order.type().equals("LOC") && sell.order.limit() == 101.5 && sell.order.qty() == 990, closed);
        Map<String, Object> st = state(state);
        Map<String, Object> p = pos(st, "XXX");
        check(Holds.num(p.get("entry")) == 100 && Holds.num(p.get("exitAbove")) == 0.012 && Holds.num(p.get("days")) == 5 && Holds.num(p.get("closes")) == 1, p);
        sell.status = "Cancelled";
        p.put("closes", 4L);
        p.put("offered", "2000-01-01");
        save(state, st);
        String last = run(b, "UK", "close", state, null);
        List<FakeBroker.T> moc = b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("SELL") && t.order.type().equals("MOC"));
        check(moc.size() == 1 && moc.get(0).order.qty() == 990 && last.contains("its last day (5)"), last);
        moc.get(0).status = "Cancelled";
        st = state(state);
        pos(st, "XXX").put("closes", 2L);
        pos(st, "XXX").put("offered", "2000-01-01");
        save(state, st);
        book.put("orders", new ArrayList<>());
        book.put("updates", List.of(upd("ticker", "XXX", "keep", true, "exitAbove", 0.02, "expected", 0.004)));
        String kept = run(b, "UK", "place", state, book);
        check(Holds.num(pos(state(state), "XXX").get("exitAbove")) == 0.02 && kept.contains("+1.20% -> +2.00%"), kept);
        book.put("updates", List.of(upd("ticker", "XXX", "keep", false, "exitAbove", 0.02, "expected", -0.001)));
        String dropped = run(b, "UK", "place", state, book);
        List<FakeBroker.T> out = b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("SELL") && t.order.ref().endsWith("-exit"));
        check(out.size() == 1 && out.get(0).order.type().equals("MKT") && out.get(0).order.qty() == 990 && dropped.contains("no longer holds it"), dropped);
        String again2 = run(b, "UK", "place", state, book);
        check(again2.contains("already sold at the open today") && b.where(t -> t.order.ref().endsWith("-exit")).size() == 1, again2);
        int n = b.where(t -> t.order.action().equals("SELL")).size();
        String closed2 = run(b, "UK", "close", state, null);
        check(b.where(t -> t.order.action().equals("SELL")).size() == n && closed2.contains("being sold since the open"), closed2);
        return placed + closed + last + kept + dropped + closed2;
    }

    static String euBook() throws Exception {
        at("Europe/Paris", 8, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> book = book("uk", 5, 5, List.of(order("PAR:MC", 500, 503, 0.01, 5, 6000, 8000)));
        String placed = run(b, "EU", "place", dir.resolve("holds-EU.json"), book);
        FakeBroker.T buy = b.find("MC", "BUY");
        ZonedDateTime u = until(buy.order.goodTill(), "Europe/Paris");
        check(buy.order.tif().equals("GTD") && buy.order.limit() == 503 && buy.order.qty() == 11 && buy.share.exchange().equals("SBF"), placed);
        check(u.getHour() == 9 && u.getMinute() == 2, u);
        at("Europe/Paris", 10, 30);
        FakeBroker b2 = new FakeBroker();
        String late = run(b2, "EU", "place", dir.resolve("holds-EU-late.json"), book);
        FakeBroker.T buy2 = b2.find("MC", "BUY");
        check(buy2.order.tif().equals("DAY") && buy2.order.limit() == 503 && late.contains("the opening auction is over"), late);
        // entryDay: a day order from the start (the opening auction, then the whole session), the same limit
        at("Europe/Paris", 8, 50);
        FakeBroker b3 = new FakeBroker();
        Map<String, Object> od = order("PAR:MC", 500, 503, 0.01, 5, 6000, 8000);
        od.put("entryDay", true);
        String day = run(b3, "EU", "place", dir.resolve("holds-EU-day.json"), book("uk", 5, 5, List.of(od)));
        FakeBroker.T buy3 = b3.find("MC", "BUY");
        check(buy3.order.tif().equals("DAY") && buy3.order.limit() == 503 && day.contains("then all day"), day);
        // the summaries name no price (they go to the public monitor's log)
        for (String x : List.of(placed, late, day)) check(!x.contains("503") && !x.contains("500"), x);
        return placed + late + day;
    }

    static String bracketBook() throws Exception {
        at("America/New_York", 9, 20);
        FakeBroker b = new FakeBroker();
        Map<String, Object> o = order("BRK.B", 480, 484, null, 1, 10000, 20000);
        o.put("bracketTake", 489.6);
        o.put("bracketStop", 460.8);
        String placed = run(b, "US", "place", dir.resolve("holds-US-bracket.json"), book("uk", 1, 5, List.of(o)));
        FakeBroker.T buy = b.find("BRK B", "BUY");
        List<FakeBroker.T> sells = b.where(t -> t.share.symbol().equals("BRK B") && t.order.action().equals("SELL"));
        check(sells.size() == 2, placed);
        FakeBroker.T take = sells.get(0), stop = sells.get(1);
        check(!buy.order.transmit() && buy.order.parentId() == 0, "the buy waits for its children: " + placed);
        check(take.order.type().equals("LMT") && take.order.limit() >= 489.6 && take.order.parentId() == buy.id && !take.order.transmit() && take.order.tif().equals("DAY"), "take-profit: " + placed);
        check(stop.order.type().equals("STP") && stop.order.aux() <= 460.8 && stop.order.parentId() == buy.id && stop.order.transmit() && stop.order.tif().equals("DAY"), "stop-loss: " + placed);
        check(take.order.ref().startsWith("ofm-USS") && stop.order.ref().startsWith("ofm-USS"), "cancelled with the stops at the close: " + placed);
        check(placed.contains("a bracket"), placed);
        return placed;
    }

    static String usBook() throws Exception {
        at("America/New_York", 9, 20);
        FakeBroker b = new FakeBroker();
        String placed = run(b, "US", "place", dir.resolve("holds-US.json"), book("uk", 5, 5, List.of(order("BRK.B", 480, 484, 0.004, 5, 10000, 20000))));
        FakeBroker.T buy = b.find("BRK B", "BUY");
        ZonedDateTime u = until(buy.order.goodTill(), "America/New_York");
        check(buy.order.tif().equals("GTD") && buy.order.limit() == 484 && buy.order.qty() == 20 && buy.share.exchange().equals("SMART"), placed);
        check(u.getHour() == 9 && u.getMinute() == 32, u);
        return placed;
    }

    static String apacBook() throws Exception {
        at("Australia/Sydney", 9, 45);
        FakeBroker b = new FakeBroker();
        String placed = run(b, "APAC", "place", dir.resolve("holds-APAC.json"), book("uk", 5, 5,
                List.of(order("ASX:BHP", 45, 45.5, 0.01, 5, 5000, 9000), order("SGX:D05", 50, 50.4, 0.01, 5, 6000, 9000))));
        ZonedDateTime bhp = until(b.find("BHP", "BUY").order.goodTill(), "Australia/Sydney");
        ZonedDateTime d05 = until(b.find("D05", "BUY").order.goodTill(), "Asia/Singapore");
        check(bhp.getHour() == 10 && bhp.getMinute() == 12 && d05.getHour() == 9 && d05.getMinute() == 2, placed);
        check(b.find("D05", "BUY").order.qty() == 100, b.find("D05", "BUY").order);
        return placed;
    }

    static String indexBook() throws Exception {
        at("Europe/Berlin", 8, 46);
        FakeBroker b = new FakeBroker();
        String placed = run(b, "INDEX", "place", dir.resolve("holds-INDEX.json"), book("uk", 5, 5, List.of(order("FRA:SPY", 600, 603, 0.003, 5, 7000, 1e9))));
        FakeBroker.T buy = b.find("SPY", "BUY");
        ZonedDateTime u = until(buy.order.goodTill(), "Europe/Berlin");
        check(buy.share.exchange().equals("IBIS") && buy.order.qty() == 11 && u.getHour() == 9 && u.getMinute() == 2, placed);
        return placed;
    }

    static String legacyPositions() throws Exception {
        at("Europe/London", 7, 45);
        FakeBroker b = new FakeBroker();
        for (String sym : List.of("AAA", "BBB")) {
            b.fills.add(new Broker.Fill(new Broker.Share(FakeBroker.CONIDS.get(sym), sym, "LSE", "LSE", "GBP"), "BOT", 10, 100.0, "ofm-UKH-2026-11-05", 0, "x" + sym, "", 0, 0));
        }
        Path state = dir.resolve("holds-UK-legacy.json");
        Map<String, Object> positions = new LinkedHashMap<>();
        positions.put("AAA", upd("conId", 1L, "qty", 10L, "entry", 100.0, "since", "2026-11-05", "closes", 2L));
        positions.put("BBB", upd("conId", 2L, "qty", 10L, "entry", 100.0, "since", "2026-11-05", "closes", 2L));
        save(state, upd("positions", positions, "pending", new LinkedHashMap<>()));
        String placed = run(b, "UK", "place", state, book("uk", 5, 2, List.of(order("XXX", 100, 101, 0.01, 5, 1000, 1000), order("DDD", 100, 101, 0.01, 5, 1000, 1000))));
        List<String> buys = new ArrayList<>();
        for (FakeBroker.T t : b.where(t -> t.order.action().equals("BUY"))) buys.add(t.share.symbol());
        buys.sort(null);
        check(buys.equals(List.of("DDD", "XXX")) && placed.contains("2 positions of an earlier book held"), buys + " " + placed);
        return placed;
    }

    static String closeOrdersRefused() throws Exception {
        at("Europe/London", 16, 20);
        FakeBroker b = new FakeBroker();
        b.noCloseOrders = Set.of("XXX");
        b.fills.add(new Broker.Fill(new Broker.Share(6, "XXX", "LSE", "LSE", "GBP"), "BOT", 990, 100.0, "ofm-UKH-" + TODAY, 0, "x1", "", 0, 0));
        Path state = dir.resolve("holds-UK-refused.json");
        Map<String, Object> pending = new LinkedHashMap<>();
        pending.put("XXX", upd("conId", 6L, "qty", 990L, "exitAbove", 0.012, "days", 5L));
        save(state, upd("positions", new LinkedHashMap<>(), "pending", pending));
        String closed = run(b, "UK", "close", state, null);
        List<FakeBroker.T> sells = b.where(t -> t.order.action().equals("SELL"));
        check(sells.size() == 2 && sells.get(0).order.type().equals("LOC") && sells.get(1).order.type().equals("LMT") && sells.get(1).order.limit() == 101.5
                && sells.get(1).order.tif().equals("DAY"), closed);
        check(sells.get(1).status.equals("Submitted") && closed.contains("for the rest of the day instead: live"), closed);
        Map<String, Object> st = state(state);
        sells.get(1).status = "Cancelled";
        pos(st, "XXX").put("closes", 4L);
        pos(st, "XXX").put("offered", "2000-01-01");
        save(state, st);
        String last = run(b, "UK", "close", state, null);
        List<FakeBroker.T> s2 = b.where(t -> t.order.action().equals("SELL"));
        check(s2.size() == 4 && s2.get(2).order.type().equals("MOC") && s2.get(3).order.type().equals("LMT") && last.contains("limit at the 20-day low instead: live"), last);
        return closed + last;
    }

    static String strategyBook() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> book = book("strategy", 260, 5, List.of(order("XXX", 100, 101, null, 260, 1000, 1000)));
        book.put("updates", List.of(upd("ticker", "XXX", "keep", true, "exitAbove", null)));
        Path state = dir.resolve("holds-UK-strategy.json");
        String placed = run(b, "UK", "place", state, book);
        b.fill(b.find("XXX", "BUY"), 990, 100.0);
        at("Europe/London", 16, 30);
        String closed = run(b, "UK", "close", state, null);
        Map<String, Object> p = pos(state(state), "XXX");
        check(p.containsKey("exitAbove") && p.get("exitAbove") == null && Holds.num(p.get("days")) == 260 && Holds.num(p.get("closes")) == 1, p);
        check(b.find("XXX", "SELL") == null && closed.contains("held (1 of at most 260 closes"), closed);
        book.put("orders", new ArrayList<>());
        book.put("updates", List.of(upd("ticker", "XXX", "keep", false, "exitAbove", null)));
        String dropped = run(b, "UK", "place", state, book);
        List<FakeBroker.T> out = b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("SELL") && t.order.ref().endsWith("-exit"));
        check(out.size() == 1 && out.get(0).order.type().equals("MKT") && out.get(0).order.qty() == 990, dropped);
        return placed + closed + dropped;
    }

    static String strategyStops() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> o = order("XXX", 100, 101, null, 260, 1000, 1000);
        o.put("stopKind", "psar");
        o.put("stopPrice", 94.0);
        Map<String, Object> book = book("strategy", 260, 5, List.of(o));
        book.put("updates", new ArrayList<>());
        Path state = dir.resolve("holds-UK-stops.json");
        String placed = run(b, "UK", "place", state, book);
        b.fill(b.find("XXX", "BUY"), 990, 100.0);
        at("Europe/London", 16, 30);
        String closed = run(b, "UK", "close", state, null);
        List<FakeBroker.T> stops = b.where(t -> t.share.symbol().equals("XXX") && t.order.type().equals("STP") && !t.status.equals("Cancelled"));
        check(stops.size() == 1 && stops.get(0).order.aux() == 94.0 && stops.get(0).order.tif().equals("GTC") && stops.get(0).order.qty() == 990, closed);
        Map<String, Object> st = state(state);
        pos(st, "XXX").put("offered", "2000-01-01");
        save(state, st);
        at("Europe/London", 7, 50);
        book.put("orders", new ArrayList<>());
        book.put("updates", List.of(upd("ticker", "XXX", "keep", true, "exitAbove", null, "stopPrice", 96.5)));
        String moved = run(b, "UK", "place", state, book);
        List<FakeBroker.T> live = b.where(t -> t.share.symbol().equals("XXX") && t.order.type().equals("STP") && !t.status.equals("Cancelled"));
        check(live.size() == 1 && live.get(0).order.aux() == 96.5, moved);
        book.put("updates", List.of(upd("ticker", "XXX", "keep", false, "exitAbove", null, "stopPrice", null)));
        String dropped = run(b, "UK", "place", state, book);
        List<FakeBroker.T> live2 = b.where(t -> t.share.symbol().equals("XXX") && t.order.type().equals("STP") && !t.status.equals("Cancelled"));
        List<FakeBroker.T> out = b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("SELL") && t.order.ref().endsWith("-exit"));
        check(live2.isEmpty() && out.size() == 1, dropped);
        return placed + closed + moved + dropped;
    }

    @SuppressWarnings("unchecked")
    static String strategyRebuy() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> o = order("XXX", 100, 101, null, 260, 1000, 1000);
        o.put("stopKind", "psar-vol");
        o.put("stopPrice", 94.0);
        o.put("rebuyStop", 0.01);
        o.put("rebuyStopRel", -0.05);
        Map<String, Object> book = book("strategy", 260, 5, List.of(o));
        book.put("updates", new ArrayList<>());
        Path state = dir.resolve("holds-UK-rebuy.json");
        String placed = run(b, "UK", "place", state, book);
        java.util.function.BiFunction<String, String, List<FakeBroker.T>> live = (kind, side) -> b.where(t -> t.share.symbol().equals("XXX")
                && t.order.type().equals(kind) && t.order.action().equals(side) && !t.status.equals("Cancelled") && !t.status.equals("Filled"));
        b.fill(b.find("XXX", "BUY"), 990, 100.0);
        at("Europe/London", 16, 30);
        String closed = run(b, "UK", "close", state, null);
        List<FakeBroker.T> stop = live.apply("STP", "SELL"), back = live.apply("STP LMT", "BUY");
        check(stop.size() == 1 && stop.get(0).order.aux() == 94.0 && stop.get(0).order.tif().equals("GTC") && !stop.get(0).order.transmit(), closed);
        check(back.size() == 1 && back.get(0).order.parentId() == stop.get(0).id && back.get(0).order.aux() == 95.0 && back.get(0).order.limit() == 95.5
                && back.get(0).order.qty() == 990 && back.get(0).order.tif().equals("GTC") && back.get(0).order.ref().startsWith("ofm-UKSB"), closed);
        at("Europe/London", 11, 0);
        b.fill(stop.get(0), 990, 94.0);
        b.fill(back.get(0), 990, 95.0);
        String prot = run(b, "UK", "protect", state, null);
        Map<String, Object> st = state(state);
        check(((Map<String, Object>) st.get("positions")).containsKey("XXX") && Holds.num(pos(st, "XXX").get("entry")) == 95.0
                && ((Map<String, Object>) st.get("stopped")).isEmpty(), st + " " + prot);
        stop = live.apply("STP", "SELL");
        back = live.apply("STP LMT", "BUY");
        check(stop.size() == 1 && stop.get(0).order.aux() == 90.0 && back.size() == 1 && back.get(0).order.aux() == 91.0 && back.get(0).order.parentId() == stop.get(0).id, prot);
        String again = run(b, "UK", "protect", state, null);
        check(!again.contains("bought back") && live.apply("STP", "SELL").size() == 1, again);
        b.fill(stop.get(0), 990, 90.0);
        at("Europe/London", 7, 50);
        book.put("updates", List.of(upd("ticker", "XXX", "keep", true, "exitAbove", null, "stopPrice", null, "rebuyStop", 0.01, "rebuyStopRel", -0.05)));
        String morning = run(b, "UK", "place", state, book);
        st = state(state);
        check(((Map<String, Object>) st.get("stopped")).containsKey("XXX") && !((Map<String, Object>) st.get("positions")).containsKey("XXX")
                && live.apply("STP LMT", "BUY").size() == 1 && live.apply("MKT", "BUY").isEmpty() && live.apply("LMT", "BUY").isEmpty()
                && morning.contains("its buy-back order works"), st + " " + morning);
        book.put("orders", new ArrayList<>());
        book.put("updates", List.of(upd("ticker", "XXX", "keep", false, "exitAbove", null, "stopPrice", null)));
        String dropped = run(b, "UK", "place", state, book);
        st = state(state);
        check(((Map<String, Object>) st.get("stopped")).isEmpty() && live.apply("STP LMT", "BUY").isEmpty() && dropped.contains("buy-back order cancelled"), dropped);
        return placed + closed + prot + morning + dropped;
    }

    /** 24 Nov: the UK with no IBKR price to check the scale against (no market data): skipped, not a NullPointerException. */
    static String ukNoPrice() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker() {
            @Override
            public double referenceClose(Share s) {
                return Double.NaN;
            }
        };
        Map<String, Object> book = book("uk", 5, 5, List.of(order("XXX", 100, 101, 0.012, 5, 1000, 1000)));
        String placed = run(b, "UK", "place", dir.resolve("holds-UK-noprice.json"), book);
        check(b.find("XXX", "BUY") == null && placed.contains("no IBKR price to check the price scale against"), placed);
        return placed;
    }

    /**
     * 24 Nov, APAC: the buys tagged with one date (the NAS's), the close run with another (Sydney's): the close found
     * none of the fills. Now the book records what IBKR holds that it bought: from the plan kept since the placing,
     * and, with the plan lost, from the fills with the book's reference.
     */
    @SuppressWarnings("unchecked")
    static String adoptAcrossDates() throws Exception {
        at("Australia/Sydney", 9, 45);
        FakeBroker b = new FakeBroker();
        Path state = dir.resolve("holds-APAC-dates.json");
        LocalDate d1 = TODAY.minusDays(1);
        String placed = run(b, "APAC", "place", state, book("uk", 5, 5, List.of(order("ASX:BHP", 45, 45.5, 0.01, 5, 5000, 9000))), d1);
        FakeBroker.T buy = b.find("BHP", "BUY");
        check(buy != null && buy.order.ref().equals("ofm-APACH-" + d1), placed);
        b.fill(buy, buy.order.qty(), 45.2);
        at("Australia/Sydney", 15, 50);
        String closed = run(b, "APAC", "close", state, null, TODAY);
        Map<String, Object> p = pos(state(state), "BHP");
        check(p != null && Holds.num(p.get("qty")) == buy.order.qty() && Holds.num(p.get("days")) == 5 && closed.contains("BHP: recorded"), closed);
        check(!((Map<String, Object>) state(state).get("plans")).containsKey("BHP"), state(state));
        // the plan lost (the state before this fix): recorded from the fills with the book's reference, any day
        Map<String, Object> st = state(state);
        ((Map<String, Object>) st.get("positions")).clear();
        st.remove("plans");
        save(state, st);
        at("Australia/Sydney", 9, 45);
        String next = run(b, "APAC", "place", state, book("uk", 5, 5, List.of()), TODAY.plusDays(1));
        p = pos(state(state), "BHP");
        check(p != null && next.contains("BHP: recorded") && next.contains("its plan lost"), next);
        // a share of the same symbol bought by hand (no plan, no fill of the book): not taken
        FakeBroker b2 = new FakeBroker();
        b2.fills.add(new Broker.Fill(new Broker.Share(FakeBroker.CONIDS.get("AAA"), "AAA", "LSE", "LSE", "GBP"), "BOT", 10, 100.0, "manual", 0, "m1", "", 0, 0));
        Path s2 = dir.resolve("holds-UK-hand.json");
        String hand = run(b2, "UK", "place", s2, book("uk", 5, 5, List.of()));
        check(((Map<String, Object>) state(s2).get("positions")).isEmpty() && !hand.contains("recorded"), hand);
        return placed + closed + next;
    }

    /** 24 Nov: IB Gateway hung through the close's window; the catch-up records the day's buy and places its stop (no offer at the close). */
    static String closeCatchUp() throws Exception {
        at("Europe/London", 7, 50);
        FakeBroker b = new FakeBroker();
        Map<String, Object> o = order("XXX", 100, 101, null, 260, 1000, 1000);
        o.put("stopKind", "psar");
        o.put("stopPrice", 94.0);
        Map<String, Object> book = book("strategy", 260, 5, List.of(o));
        book.put("updates", new ArrayList<>());
        Path state = dir.resolve("holds-UK-catchup.json");
        String placed = run(b, "UK", "place", state, book);
        b.fill(b.find("XXX", "BUY"), 990, 100.0);
        at("Europe/London", 19, 0);
        String rec = run(b, "UK", "record", state, null);
        Map<String, Object> p = pos(state(state), "XXX");
        List<FakeBroker.T> stops = b.where(t -> t.share.symbol().equals("XXX") && t.order.type().equals("STP") && !t.status.equals("Cancelled"));
        List<FakeBroker.T> offers = b.where(t -> t.share.symbol().equals("XXX") && t.order.action().equals("SELL") && !t.order.type().equals("STP"));
        check(p != null && Holds.num(p.get("qty")) == 990 && Holds.num(p.get("closes")) == 0 && stops.size() == 1 && stops.get(0).order.aux() == 94.0
                && stops.get(0).order.tif().equals("GTC") && offers.isEmpty() && ((Map<?, ?>) state(state).get("pending")).isEmpty(), rec);
        String again = run(b, "UK", "record", state, null);
        List<FakeBroker.T> stops2 = b.where(t -> t.share.symbol().equals("XXX") && t.order.type().equals("STP") && !t.status.equals("Cancelled"));
        check(stops2.size() == 1 && again.contains("its stop kept"), again);
        return placed + rec + again;
    }

    public static void main(String[] args) throws Exception {
        dir = Files.createTempDirectory("holds-check");
        System.out.print(closeCatchUp());
        System.out.println("the close missed: caught up in the evening (the day's buy recorded, its stop placed, nothing offered; kept when run again); OK");
        System.out.print(adoptAcrossDates());
        System.out.println("APAC across dates: what IBKR holds that the book bought is recorded (from its plan, or its fills); a share bought by hand is not; OK");
        System.out.print(ukNoPrice());
        System.out.println("UK without IBKR's price: the buy skipped with its reason; OK");
        System.out.print(holdBook());
        System.out.println("hold book: bought in the opening auction only; offered at each close at or above its exit level; sold at any price on its last day; "
                + "updated every morning (kept with its new exit level, or sold at the open); OK");
        System.out.print(strategyBook());
        System.out.println("strategy book: held with no exit level, sold at the open when the strategy drops it; OK");
        System.out.print(strategyStops());
        System.out.println("strategy book: a GTC stop at the PSAR level once bought, moved up each morning, cancelled before the sale at the open; OK");
        System.out.print(strategyRebuy());
        System.out.println("strategy book: the buy-back coupled to the stop, a position again with its own pair once filled, cancelled when the strategy drops it; OK");
        System.out.print(euBook());
        System.out.println("Europe: an EXCHANGE:CODE order in euros, good only through Paris's opening auction; OK");
        System.out.print(bracketBook());
        System.out.println("intraday bracket: the buy the parent, a take-profit limit and a stop its day-order children, the last transmitting all; OK");
        System.out.print(usBook());
        System.out.println("US: BRK.B as BRK B, routed SMART, good only through New York's opening auction; OK");
        System.out.print(apacBook());
        System.out.println("Asia-Pacific: ASX until 10:12 Sydney, SGX until 09:02 Singapore in round lots; OK");
        System.out.print(indexBook());
        System.out.println("INDEX: a Xetra ETF on IBIS, good only through Frankfurt's opening auction; OK");
        System.out.print(legacyPositions());
        System.out.println("hold book: an earlier book's positions do not take the live book's slots; OK");
        System.out.print(closeOrdersRefused());
        System.out.println("hold book: where the exchange takes no LOC or MOC, a day limit at the exit level (the last day: at the 20-day low); OK");
    }
}
