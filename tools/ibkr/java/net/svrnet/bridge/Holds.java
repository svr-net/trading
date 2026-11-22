package net.svrnet.bridge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Clock;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * The multi-day book (submit.py --holds place|protect|close, ported line by line): its own orders (orderRef
 * ofm-<MARKET>H-<date>) and its positions kept in the hold-state file (the bridge's history folder).
 *   place    the positions no longer held are dropped (sold); the daily update sells what the model no longer holds
 *            and moves the others' stops; each order for a share not held is placed (a limit buy at its limit, or in
 *            the opening auction only: entry "open", good till a few minutes after the open)
 *   protect  a buy-back filled since the last run: a position again with its own stop and buy-back at once
 *   close    today's fills become positions; today's unfilled buys are cancelled; each position is offered in the
 *            closing auction (its last day at any price; before that at or above its exit level), its stop kept
 * A stop: a good-till-cancelled stop sell at IBKR (orderRef ofm-<MARKET>S-...), with its buy-back coupled to it as a
 * child (a buy stop-limit, or a trailing one), moved each morning to the day's level.
 */
final class Holds {
    static final Map<String, String[]> VENUE = Map.of("UK", new String[] {"LSE", "GBP"}, "US", new String[] {"SMART", "USD"},
            "APAC", new String[] {"ASX", "AUD"}, "EU", new String[] {"SBF", "EUR"}, "INDEX", new String[] {"IBIS", "EUR"});
    static final Map<String, String[]> EXCHANGE = Map.ofEntries(
            Map.entry("AMEX", new String[] {"SMART", "USD"}), Map.entry("ARCA", new String[] {"SMART", "USD"}), Map.entry("NYSE", new String[] {"SMART", "USD"}),
            Map.entry("NASDAQ", new String[] {"SMART", "USD"}), Map.entry("BATS", new String[] {"SMART", "USD"}), Map.entry("PAR", new String[] {"SBF", "EUR"}),
            Map.entry("AMS", new String[] {"AEB", "EUR"}), Map.entry("BRU", new String[] {"ENEXT.BE", "EUR"}), Map.entry("LIS", new String[] {"BVL", "EUR"}),
            Map.entry("MSE", new String[] {"BM", "EUR"}), Map.entry("FRA", new String[] {"IBIS", "EUR"}), Map.entry("OSL", new String[] {"OSE", "NOK"}),
            Map.entry("ASX", new String[] {"ASX", "AUD"}), Map.entry("SGX", new String[] {"SGX", "SGD"}));
    static final Map<String, Object[]> OPEN = Map.of("UK", new Object[] {"Europe/London", 8, 0}, "EU", new Object[] {"Europe/Paris", 9, 0},
            "US", new Object[] {"America/New_York", 9, 30}, "APAC", new Object[] {"Australia/Sydney", 10, 0}, "INDEX", new Object[] {"Europe/Berlin", 9, 0});
    static final Map<String, Object[]> AUCTION_END = Map.of("ASX", new Object[] {"Australia/Sydney", 10, 12}, "SGX", new Object[] {"Asia/Singapore", 9, 2});
    static final int AUCTION_MINUTES = 2;
    static final Map<String, Integer> BOARD_LOT = Map.of("SGX", 100);
    /** The clock (the checks set their own day and time). */
    static Clock clock = Clock.systemUTC();

    // --- submit.py's helpers ------------------------------------------------------------------------------------
    static String ibSymbol(String ticker, String market) {
        return market.equals("US") ? ticker.replace('.', ' ') : ticker;
    }

    static Broker.Share shareOf(String ticker, String market) {
        int k = ticker.indexOf(':');
        if (k >= 0) {
            String[] v = EXCHANGE.get(ticker.substring(0, k));
            if (v == null) return null;
            return new Broker.Share(0, ticker.substring(k + 1), v[0], "", v[1]);
        }
        String[] v = VENUE.get(market);
        return new Broker.Share(0, ibSymbol(ticker, market), v[0], "", v[1]);
    }

    static double roundTo(double x, double tick, String how) {
        double n = x / tick;
        n = how.equals("down") ? Math.floor(n + 1e-9) : how.equals("up") ? Math.ceil(n - 1e-9) : Math.rint(n);
        return Math.round(n * tick * 1e10) / 1e10;
    }

    static double num(Object v) {
        return v instanceof Number ? ((Number) v).doubleValue() : v instanceof String && !((String) v).isEmpty() ? Double.parseDouble((String) v) : Double.NaN;
    }

    static boolean has(Map<String, Object> m, String k) {
        return m.get(k) != null;
    }

    static String g(double x) {  // Python's {:g}
        if (Double.isNaN(x)) return "nan";
        String s = String.format(Locale.ROOT, "%.6g", x);
        if (s.contains("e")) {
            String[] p = s.split("e");
            String m = p[0].contains(".") ? p[0].replaceAll("0+$", "").replaceAll("\\.$", "") : p[0];
            int e = Integer.parseInt(p[1]);
            return m + "e" + (e < 0 ? "-" : "+") + String.format(Locale.ROOT, "%02d", Math.abs(e));
        }
        return s.contains(".") ? s.replaceAll("0+$", "").replaceAll("\\.$", "") : s;
    }

    static double sharesFor(Map<String, Object> o, double money, double price) {
        if (!(price > 0)) return 0;
        String t = String.valueOf(o.getOrDefault("ticker", ""));
        String ex = t.contains(":") ? t.substring(0, t.indexOf(':')) : "";
        int lot = BOARD_LOT.getOrDefault(ex, 1);
        return Math.floor(money / price / lot + 1e-9) * lot;
    }

    static double capitalOf(Map<String, Object> o, double capital) {
        double c = has(o, "capital") && num(o.get("capital")) != 0 ? num(o.get("capital")) : capital;
        return has(o, "maxCapital") && num(o.get("maxCapital")) != 0 ? Math.min(c, num(o.get("maxCapital"))) : c;
    }

    static double perShare(Map<String, Object> o, String market) {
        return market.equals("UK") ? num(o.get("limit")) / 100.0 : num(o.get("limit"));
    }

    static String auctionEndUtc(String market, String ticker) {
        DateTimeFormatter f = DateTimeFormatter.ofPattern("yyyyMMdd-HH:mm:ss");
        String ex = ticker.contains(":") ? ticker.substring(0, ticker.indexOf(':')) : "";
        Object[] a = AUCTION_END.get(ex);
        if (a != null) {
            ZonedDateTime t = ZonedDateTime.now(clock.withZone(ZoneId.of((String) a[0]))).withHour((int) a[1]).withMinute((int) a[2]).withSecond(0).withNano(0);
            return t.withZoneSameInstant(ZoneOffset.UTC).format(f);
        }
        Object[] o = OPEN.get(market);
        ZonedDateTime opens = ZonedDateTime.now(clock.withZone(ZoneId.of((String) o[0]))).withHour((int) o[1]).withMinute((int) o[2]).withSecond(0).withNano(0);
        return opens.plusMinutes(AUCTION_MINUTES).withZoneSameInstant(ZoneOffset.UTC).format(f);
    }

    static String said(List<String> msgs) {
        return msgs.isEmpty() ? "" : " [IBKR: " + String.join(" | ", msgs) + "]";
    }

    /** sell_out: at the close (MOC) or at the next price; a limit at the 20-day low when that does not survive. */
    static String[] sellOut(Broker ib, Broker.Share share, double qty, String ref, boolean atClose) {
        Broker.Share c = ib.exitShare(share);
        if (c == null) return new String[] {"false", share.symbol() + ": not found at IBKR, NOT SOLD"};
        String kind = atClose ? "at the close" : "at market";
        Broker.Fate f = ib.fate(ib.place(c, Broker.NewOrder.of("SELL", atClose ? "MOC" : "MKT", qty, 0, 0, ref, "DAY")), 20);
        String s1 = said(f.messages());
        if (!f.fate().equals("dead")) return new String[] {"true", c.symbol() + ": sell " + g(qty) + " " + kind + ": " + f.fate() + s1};
        double lim = ib.floorPrice(c);
        if (Double.isNaN(lim)) return new String[] {"false", c.symbol() + ": sell " + kind + " did not survive" + s1 + "; no price for a limit: NOT SOLD"};
        Broker.Fate f2 = ib.fate(ib.place(c, Broker.NewOrder.of("SELL", "LMT", qty, lim, 0, ref, "DAY")), 20);
        String s2 = said(f2.messages());
        if (f2.fate().equals("dead")) return new String[] {"false", c.symbol() + ": sell " + kind + " did not survive" + s1 + "; limit at the 20-day low did not survive either" + s2 + ": NOT SOLD"};
        return new String[] {"true", c.symbol() + ": sell " + kind + " did not survive" + s1 + "; limit at the 20-day low instead: " + f2.fate() + s2};
    }

    // --- the book -------------------------------------------------------------------------------------------------
    private final Broker ib;
    private final String market;
    private final String action;
    private final Path statePath;
    private final double capital;
    private final Map<String, Object> st;
    private final String stopRef;
    private final String ref;
    private final String today;
    private final List<String> lines = new ArrayList<>();
    private Map<String, Broker.Position> held;

    Holds(Broker ib, String market, String action, Path statePath, double capital, LocalDate today) throws IOException {
        this.ib = ib;
        this.market = market;
        this.action = action;
        this.statePath = statePath;
        this.capital = capital;
        this.today = today.toString();
        this.stopRef = "ofm-" + market + "S";
        this.ref = "ofm-" + market + "H-" + this.today;
        Map<String, Object> s = statePath != null && Files.exists(statePath)
                ? JsonParse.object(Files.readString(statePath, StandardCharsets.UTF_8)) : new LinkedHashMap<>();
        s.putIfAbsent("positions", new LinkedHashMap<>());
        s.putIfAbsent("pending", new LinkedHashMap<>());
        s.putIfAbsent("stopped", new LinkedHashMap<>());
        this.st = s;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(String k) {
        return (Map<String, Object>) st.get(k);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object v) {
        return v instanceof Map ? (Map<String, Object>) v : new LinkedHashMap<>();
    }

    private List<Broker.Ord> stopsOf(String sym) {
        List<Broker.Ord> r = new ArrayList<>();
        for (Broker.Ord t : ib.openOrders()) if (t.share().symbol().equals(sym) && t.ref() != null && t.ref().startsWith(stopRef) && !t.dead()) r.add(t);
        return r;
    }

    private void cancelStops(String sym) {
        for (Broker.Ord t : stopsOf(sym)) ib.cancel(t.orderId());
    }

    private List<Broker.Ord> rebuysOf(String sym) {
        List<Broker.Ord> r = new ArrayList<>();
        for (Broker.Ord t : stopsOf(sym)) if (t.action().equals("BUY")) r.add(t);
        return r;
    }

    private static boolean sameBack(Broker.Ord o, double back, double rebuy, boolean trail) {
        if (trail) return o.type().equals("TRAIL LIMIT") && Math.abs(o.trailPercent() - Math.round(100 * rebuy * 100) / 100.0) < 1e-9;
        return !o.type().equals("TRAIL LIMIT") && Math.abs(o.aux() - back) < 1e-9;
    }

    /** set_stop: the position's stop at `level` with (rebuy) its buy-back coupled to it; kept when both are there. */
    String setStop(String sym, Broker.Share share, double qty, double level, Object rebuyObj, Object trailObj) {
        Broker.Share c = share == null ? null : ib.exitShare(share);
        if (c == null) return sym + ": not found at IBKR, NO STOP";
        Broker.Ticks t = ib.ticks(c);
        boolean trail = Boolean.TRUE.equals(trailObj);
        double rebuy = rebuyObj == null ? 0 : num(rebuyObj);
        double lim = roundTo(level, t.at(level), "down");
        Double back = rebuy != 0 ? roundTo(lim * (1 + rebuy), t.at(lim * (1 + rebuy)), "up") : null;
        List<Broker.Ord> cur = stopsOf(sym);
        List<Broker.Ord> sells = new ArrayList<>(), buys = new ArrayList<>();
        for (Broker.Ord o : cur) (o.action().equals("SELL") ? sells : buys).add(o);
        if (sells.size() == 1 && Math.abs(sells.get(0).aux() - lim) < 1e-9 && sells.get(0).qty() == qty
                && (back == null && buys.isEmpty() || back != null && buys.size() == 1 && sameBack(buys.get(0), back, rebuy, trail))) {
            return sym + ": its stop kept at " + g(lim) + (back != null ? ", its buy-back at " + g(back) : "");
        }
        cancelStops(sym);
        Broker.NewOrder stop = Broker.NewOrder.of("SELL", "STP", qty, 0, lim, stopRef + "-" + today, "GTC");
        if (back != null) stop = stop.held();
        int stopId = ib.place(c, stop);
        if (back == null) {
            Broker.Fate f = ib.fate(stopId, 20);
            return sym + ": stop at " + g(lim) + " (good till cancelled): " + f.fate() + said(f.messages());
        }
        double cap = roundTo(back * 1.01, t.at(back * 1.01), "down");
        Broker.NewOrder child;
        if (trail) {
            double off = roundTo(back * 0.01, t.at(back), "down");
            if (off == 0) off = t.at(back);
            child = new Broker.NewOrder("BUY", "TRAIL LIMIT", qty, 0, 0, stopRef + "B-" + today, "GTC", null, stopId, true,
                    Math.round(100 * rebuy * 100) / 100.0, back, off);
        } else {
            child = new Broker.NewOrder("BUY", "STP LMT", qty, cap, back, stopRef + "B-" + today, "GTC", null, stopId, true, 0, 0, 0);
        }
        Broker.Fate f = ib.fate(ib.place(c, child), 20);
        Broker.Fate f0 = ib.fate(stopId, 20);
        List<String> msgs = new ArrayList<>(f0.messages());
        msgs.addAll(f.messages());
        String how = trail ? String.format(Locale.ROOT, "trailing %.2f%% over the lowest price after the stop, starting at %s", 100 * rebuy, g(back))
                : "at " + g(back) + " (at most " + g(cap) + ")";
        return sym + ": stop at " + g(lim) + " with its buy-back " + how + " coupled to it, good till cancelled: " + f0.fate() + "/" + f.fate() + said(msgs);
    }

    private Broker.Share planShare(Map<String, Object> plan) {
        Object id = plan.get("conId");
        return id == null ? null : new Broker.Share((int) num(id), "", "SMART", "", "");
    }

    /** bought_back: a share bought again by its buy-back order: a position again, its stop from the price paid. */
    private String boughtBack(String sym, double qty, double entry, String why) {
        Map<String, Object> plan = asMap(map("stopped").remove(sym));
        Map<String, Object> pos = new LinkedHashMap<>(plan);
        pos.remove("stoppedOn");
        pos.put("qty", qty);
        pos.put("entry", entry);
        pos.put("since", today);
        pos.put("closes", 0L);
        pos.remove("offered");
        pos.remove("exiting");
        Object rel = plan.get("rebuyStopRel");
        pos.put("stop", rel != null ? entry * (1 + num(rel)) : 0.95 * entry);
        map("positions").put(sym, pos);
        return sym + ": bought back " + g(qty) + " at " + g(entry) + " (" + why + "); "
                + setStop(sym, planShare(plan), qty, num(pos.get("stop")), plan.get("rebuyStop"), plan.get("rebuyTrail"));
    }

    @SuppressWarnings("unchecked")
    private void protect() {
        List<Object> execs = st.get("rebuyExecs") instanceof List ? (List<Object>) st.get("rebuyExecs") : new ArrayList<>();
        st.put("rebuyExecs", execs);
        Set<Object> seen = new LinkedHashSet<>(execs);
        Map<String, double[]> got = new LinkedHashMap<>();
        Map<String, List<String>> ids = new HashMap<>();
        for (Broker.Fill f : ib.fills()) {
            if (f.ref() != null && f.ref().startsWith(stopRef + "B") && f.side().equals("BOT") && !seen.contains(f.execId())) {
                double[] v = got.computeIfAbsent(f.share().symbol(), k -> new double[2]);
                v[0] += f.shares();
                v[1] += f.shares() * f.price();
                ids.computeIfAbsent(f.share().symbol(), k -> new ArrayList<>()).add(f.execId());
            }
        }
        for (Map.Entry<String, double[]> e : got.entrySet()) {
            String sym = e.getKey();
            double q = e.getValue()[0], v = e.getValue()[1];
            boolean partly = false;
            for (Broker.Ord t : rebuysOf(sym)) if (t.filled() > 0 && t.remaining() > 0) partly = true;
            if (partly) {
                lines.add(sym + ": its buy-back partly filled (" + g(q) + " so far): taken when complete");
                continue;
            }
            if (map("positions").containsKey(sym) && !map("stopped").containsKey(sym)) {
                Map<String, Object> was = new LinkedHashMap<>(asMap(map("positions").remove(sym)));
                was.put("stoppedOn", today);
                map("stopped").put(sym, was);
            }
            if (map("stopped").containsKey(sym)) lines.add(boughtBack(sym, q, v / q, "its buy-back order filled"));
            List<Object> all = new ArrayList<>(execs);
            all.addAll(ids.get(sym));
            st.put("rebuyExecs", new ArrayList<>(all.subList(Math.max(0, all.size() - 500), all.size())));
            execs = (List<Object>) st.get("rebuyExecs");
        }
    }

    /** Runs the step; returns submit.py's last line (the step's summary). */
    @SuppressWarnings("unchecked")
    String run(Map<String, Object> book) throws IOException {
        held = new LinkedHashMap<>();
        for (Broker.Position p : ib.positions()) if (p.qty() > 0) held.put(p.share().symbol(), p);
        if (action.equals("protect")) protect();
        if (action.equals("place")) place(book == null ? new LinkedHashMap<>() : book);
        else if (action.equals("close")) close();
        ib.sleep(2);
        save();
        return market + " multi-day book (" + action + "): " + map("positions").size() + " held" + (lines.isEmpty() ? "" : ": " + String.join("; ", lines));
    }

    List<String> lines() {
        return lines;
    }

    void save() throws IOException {
        if (statePath == null) return;
        Path tmp = Path.of(statePath + ".tmp");
        Files.writeString(tmp, Json.write(st), StandardCharsets.UTF_8);
        Files.move(tmp, statePath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    @SuppressWarnings("unchecked")
    private void place(Map<String, Object> book) {
        protect();
        for (String sym : new ArrayList<>(map("stopped").keySet())) {
            if (held.containsKey(sym) && !map("positions").containsKey(sym)) {
                Broker.Position h = held.get(sym);
                double entry = h.avgCost() != 0 ? h.avgCost() : num(asMap(map("stopped").get(sym)).getOrDefault("entry", 0.0));
                lines.add(boughtBack(sym, h.qty(), entry, "held again"));
            }
        }
        for (String sym : new ArrayList<>(map("positions").keySet())) {
            if (held.containsKey(sym)) continue;
            Map<String, Object> pos = asMap(map("positions").remove(sym));
            boolean working = false;
            for (Broker.Ord t : rebuysOf(sym)) if (!t.dead()) working = true;
            if (pos.get("rebuyStop") != null && num(pos.get("rebuyStop")) != 0 && working) {
                Map<String, Object> was = new LinkedHashMap<>(pos);
                was.put("stoppedOn", today);
                map("stopped").put(sym, was);
                lines.add(sym + ": sold by its stop; its buy-back order works");
            } else {
                lines.add(sym + ": sold (no longer held)");
            }
        }
        Map<String, Object> before = asMap(st.get("pending"));
        st.put("pending", new LinkedHashMap<>());
        Set<String> working = new LinkedHashSet<>();
        for (Broker.Ord t : ib.openOrders()) if (ref.equals(t.ref()) && t.action().equals("BUY") && !t.dead()) working.add(t.share().symbol());
        Map<String, Map<String, Object>> upd = new LinkedHashMap<>();
        Object ups = book.get("updates");
        if (ups instanceof List) {
            for (Object x : (List<Object>) ups) {
                Map<String, Object> u = asMap(x);
                Object t = u.get("ticker");
                if (t == null || String.valueOf(t).isEmpty()) continue;
                String tk = String.valueOf(t);
                upd.put(ibSymbol(tk.substring(tk.lastIndexOf(':') + 1), market), u);
            }
        }
        for (String sym : new ArrayList<>(map("stopped").keySet())) {  // a stopped share's buy-back works while the strategy still wants it
            Map<String, Object> u = upd.get(sym);
            if (u == null || Boolean.FALSE.equals(u.getOrDefault("keep", true))) {
                for (Broker.Ord t : rebuysOf(sym)) ib.cancel(t.orderId());
                map("stopped").remove(sym);
                lines.add(sym + ": the strategy no longer wants it: its buy-back order cancelled");
            } else if (u.get("rebuyStopRel") != null) {
                asMap(map("stopped").get(sym)).put("rebuyStopRel", num(u.get("rebuyStopRel")));
            }
        }
        for (Map.Entry<String, Object> e : map("positions").entrySet()) {
            String sym = e.getKey();
            Map<String, Object> pos = asMap(e.getValue());
            Map<String, Object> u = upd.get(sym);
            if (u == null || !held.containsKey(sym)) continue;
            Broker.Position h = held.get(sym);
            if (Boolean.FALSE.equals(u.getOrDefault("keep", true))) {
                if (today.equals(pos.get("exiting"))) {
                    lines.add(sym + ": already sold at the open today");
                    continue;
                }
                double qty = Math.min(pos.get("qty") != null ? num(pos.get("qty")) : h.qty(), h.qty());
                cancelStops(sym);
                String[] r = sellOut(ib, h.share(), qty, ref + "-exit", false);
                if (r[0].equals("true")) pos.put("exiting", today);
                lines.add(sym + ": the model no longer holds it (" + String.format(Locale.ROOT, "%+.2f", 100 * (u.get("expected") == null ? 0 : num(u.get("expected"))))
                        + "% expected), sold at the open: " + r[1]);
            } else if (u.get("stopPrice") != null && pos.get("scale") != null && num(pos.get("scale")) != 0) {
                pos.put("stop", num(u.get("stopPrice")) * num(pos.get("scale")));
                if (u.get("rebuyStop") != null) pos.put("rebuyStop", num(u.get("rebuyStop")));
                if (u.get("rebuyStopRel") != null) pos.put("rebuyStopRel", num(u.get("rebuyStopRel")));
                if (u.get("rebuyTrail") != null) pos.put("rebuyTrail", Boolean.TRUE.equals(u.get("rebuyTrail")));
                double qty = Math.min(pos.get("qty") != null ? num(pos.get("qty")) : h.qty(), h.qty());
                lines.add(setStop(sym, h.share(), qty, num(pos.get("stop")), pos.get("rebuyStop"), pos.get("rebuyTrail")));
            } else if (u.get("exitAbove") != null && Math.abs(num(u.get("exitAbove")) - (pos.get("exitAbove") == null ? 0 : num(pos.get("exitAbove")))) > 1e-9) {
                lines.add(sym + ": kept; its exit level " + String.format(Locale.ROOT, "%+.2f%% -> %+.2f%%", 100 * (pos.get("exitAbove") == null ? 0 : num(pos.get("exitAbove"))),
                        100 * num(u.get("exitAbove"))) + " above the price paid");
                pos.put("exitAbove", num(u.get("exitAbove")));
            }
        }
        // The book's slots not held: its own positions only (positions with no plan of their own: an earlier book's).
        int own = 0;
        for (Object p : map("positions").values()) if (asMap(p).containsKey("days")) own++;
        long slots = book.get("slots") instanceof Number ? ((Number) book.get("slots")).longValue() : 10;
        long free = Math.max(0, slots - own);
        if (own < map("positions").size()) {
            lines.add((map("positions").size() - own) + " positions of an earlier book held (sold by their own rule); " + free + " of the book's " + slots + " slots free");
        }
        Object os = book.get("orders");
        for (Object x : os instanceof List ? (List<Object>) os : List.of()) {
            Map<String, Object> o = asMap(x);
            String ticker = String.valueOf(o.get("ticker"));
            String sym = ibSymbol(ticker.substring(ticker.lastIndexOf(':') + 1), market);
            if (free <= 0) {
                lines.add(sym + ": every slot of the book held, not bought");
                continue;
            }
            if (working.contains(sym)) {
                if (before.containsKey(sym)) map("pending").put(sym, before.get(sym));
                free--;
                lines.add(sym + ": its buy of today is already working at IBKR, kept");
                continue;
            }
            if (held.containsKey(sym) || map("positions").containsKey(sym)) {
                lines.add(sym + ": held already or in the day's orders, not bought");
                continue;
            }
            if (map("stopped").containsKey(sym)) {
                lines.add(sym + ": its buy-back order works since its stop, not bought at the open");
                continue;
            }
            Broker.Share c0 = shareOf(ticker, market);
            Broker.Share c = c0 == null ? null : ib.qualify(c0);
            if (c == null) {
                lines.add(sym + ": not found at IBKR");
                continue;
            }
            double px = has(o, "limit") && num(o.get("limit")) != 0 ? num(o.get("limit")) : num(o.get("close"));
            double qty = 0;
            if (px > 0) {
                Map<String, Object> withPx = new LinkedHashMap<>(o);
                withPx.put("limit", px);
                qty = sharesFor(o, capitalOf(o, capital), perShare(withPx, market));
            }
            if (qty < 1) {
                lines.add(sym + ": the amount buys no share");
                continue;
            }
            Broker.NewOrder order;
            String how;
            double scale = 1.0;
            boolean openEntry = "open".equals(o.get("entry")) && has(o, "limit") && has(o, "close");
            if (openEntry || (!Boolean.TRUE.equals(o.get("open")) && has(o, "limit") && num(o.get("limit")) != 0)) {
                double ibkrClose = ib.referenceClose(c);
                Double r = !Double.isNaN(ibkrClose) ? ibkrClose / num(o.get("close")) : (market.equals("UK") ? null : 1.0);
                Double sc = null;
                if (r != null && r > 0) for (double s : new double[] {1.0, 0.01, 100.0}) if (Math.abs(Math.log(r / s)) < 0.05) { sc = s; break; }
                Broker.Share c2 = ib.exitShare(c);
                if (sc == null || c2 == null) {
                    lines.add(sym + ": no IBKR price to check the price scale against; skipped");
                    continue;
                }
                scale = sc;
                Broker.Ticks t = ib.ticks(c2);
                double lim = roundTo(num(o.get("limit")) * scale, t.at(num(o.get("limit")) * scale), "down");
                if (openEntry) {  // the auction only, at most its limit
                    order = Broker.NewOrder.of("BUY", "LMT", qty, lim, 0, ref, "GTD").until(auctionEndUtc(market, ticker));
                    how = "in the opening auction, at most " + g(lim);
                    if (order.goodTill().compareTo(ZonedDateTime.now(clock.withZone(ZoneOffset.UTC)).format(DateTimeFormatter.ofPattern("yyyyMMdd-HH:mm:ss"))) <= 0) {
                        order = Broker.NewOrder.of("BUY", "LMT", qty, lim, 0, ref, "DAY");  // placed by hand after the auction
                        how = "at most " + g(lim) + " (a day order: the opening auction is over)";
                    }
                } else {
                    order = Broker.NewOrder.of("BUY", "LMT", qty, lim, 0, ref, "DAY");
                    how = "at " + g(lim);
                }
            } else {
                order = Broker.NewOrder.of("BUY", "MKT", qty, 0, 0, ref, "DAY");
                how = "in the opening auction";
            }
            int id = ib.place(c, order);
            Broker.Fate f = ib.fate(id, 20);
            if (f.fate().equals("dead") && "GTD".equals(order.tif()) && f.messages().stream().anyMatch(m -> m.contains("GTD") || m.toLowerCase(Locale.ROOT).contains("time in force") || m.contains("TIF"))) {
                f = ib.fate(ib.place(c, Broker.NewOrder.of("BUY", "LMT", qty, order.limit(), 0, ref, "DAY")), 20);
                how += " (a day order: IBKR refused the good-till time)";
            }
            Map<String, Object> plan = new LinkedHashMap<>();
            plan.put("conId", (long) c.conId());
            plan.put("qty", qty);
            plan.put("exit", o.getOrDefault("exit", "up"));
            plan.put("exitAbove", o.get("exitAbove") == null ? null : num(o.get("exitAbove")));
            plan.put("days", o.get("days") == null || num(o.get("days")) == 0 ? 5L : (long) num(o.get("days")));
            if (o.get("stopPrice") != null && "open".equals(o.get("entry"))) {
                plan.put("stopPrice", num(o.get("stopPrice")) * scale);
                plan.put("scale", scale);
            }
            if (o.get("rebuyStop") != null) {
                plan.put("rebuyStop", num(o.get("rebuyStop")));
                plan.put("rebuyStopRel", o.get("rebuyStopRel") == null ? null : num(o.get("rebuyStopRel")));
                plan.put("rebuyTrail", Boolean.TRUE.equals(o.get("rebuyTrail")));
            }
            map("pending").put(sym, plan);
            free--;
            lines.add(sym + ": buy " + (long) qty + " " + how + ": " + f.fate() + said(f.messages()));
        }
    }

    private void close() {
        protect();
        Map<String, double[]> got = new LinkedHashMap<>();
        Map<String, Broker.Share> gotShare = new HashMap<>();
        for (Broker.Fill f : ib.fills()) {
            if (ref.equals(f.ref()) && f.side().equals("BOT")) {
                double[] v = got.computeIfAbsent(f.share().symbol(), k -> new double[2]);
                v[0] += f.shares();
                v[1] += f.shares() * f.price();
                gotShare.putIfAbsent(f.share().symbol(), f.share());
            }
        }
        for (Broker.Ord t : ib.openOrders()) if (ref.equals(t.ref()) && t.action().equals("BUY") && !t.dead()) ib.cancel(t.orderId());
        Map<String, Object> pend = asMap(st.get("pending"));
        for (Map.Entry<String, double[]> e : got.entrySet()) {
            String sym = e.getKey();
            double q = e.getValue()[0], v = e.getValue()[1];
            if (today.equals(asMap(map("positions").get(sym)).get("since"))) continue;  // recorded by an earlier run today
            Map<String, Object> plan = asMap(pend.get(sym));
            Map<String, Object> pos = new LinkedHashMap<>();
            pos.put("conId", (long) gotShare.get(sym).conId());
            pos.put("qty", q);
            pos.put("entry", v / q);
            pos.put("since", today);
            pos.put("closes", 0L);
            pos.put("exitAbove", plan.containsKey("exitAbove") && plan.get("exitAbove") == null ? null : (plan.get("exitAbove") == null ? 0.0 : num(plan.get("exitAbove"))));
            pos.put("days", plan.get("days") == null || num(plan.get("days")) == 0 ? 5L : (long) num(plan.get("days")));
            if (plan.get("stopPrice") != null) {
                double lvl = num(plan.get("stopPrice"));
                pos.put("stop", lvl < v / q ? lvl : 0.95 * v / q);
                pos.put("scale", plan.get("scale") == null || num(plan.get("scale")) == 0 ? 1.0 : num(plan.get("scale")));
            }
            if (plan.get("rebuyStop") != null) {
                pos.put("rebuyStop", plan.get("rebuyStop"));
                pos.put("rebuyStopRel", plan.get("rebuyStopRel"));
                pos.put("rebuyTrail", Boolean.TRUE.equals(plan.get("rebuyTrail")));
            }
            map("positions").put(sym, pos);
            lines.add(sym + ": bought " + g(q) + " at " + g(v / q));
        }
        st.put("pending", new LinkedHashMap<>());
        for (Map.Entry<String, Object> e : map("positions").entrySet()) {
            String sym = e.getKey();
            if (!held.containsKey(sym)) continue;
            Map<String, Object> pos = asMap(e.getValue());
            Broker.Position p = held.get(sym);
            double qty = Math.min(pos.get("qty") != null ? num(pos.get("qty")) : p.qty(), p.qty());
            if (today.equals(pos.get("offered"))) {
                lines.add(sym + ": already offered at this close");
                continue;
            }
            if (today.equals(pos.get("exiting"))) {
                lines.add(sym + ": being sold since the open (the daily update): not offered at the close");
                continue;
            }
            long closes = (pos.get("closes") == null ? 0 : (long) num(pos.get("closes"))) + 1;
            pos.put("closes", closes);
            pos.put("offered", today);
            Broker.Share c = ib.exitShare(p.share());
            if (c == null) {
                lines.add(sym + ": not found at IBKR, NOT OFFERED");
                continue;
            }
            Broker.Ticks ticks = ib.ticks(c);
            long last = pos.get("days") == null || num(pos.get("days")) == 0 ? 5 : (long) num(pos.get("days"));
            if (pos.get("stop") != null && num(pos.get("stop")) != 0 && closes < last) {
                lines.add(setStop(sym, p.share(), qty, num(pos.get("stop")), pos.get("rebuyStop"), pos.get("rebuyTrail")));
            }
            if (closes >= last) {
                cancelStops(sym);
                String[] r = sellOut(ib, c, qty, ref + "-close", true);
                lines.add(sym + ": its last day (" + last + "): " + r[1]);
                continue;
            }
            if (pos.containsKey("exitAbove") && pos.get("exitAbove") == null) {
                lines.add(sym + ": held (" + closes + " of at most " + last + " closes; sold when the strategy drops it)");
                continue;
            }
            double entry = num(pos.get("entry"));
            double tick = ticks.at(entry);
            double level = entry * (1 + (pos.get("exitAbove") == null ? 0 : num(pos.get("exitAbove"))));
            double lim = Math.max(roundTo(entry + tick, tick, "up"), roundTo(level, ticks.at(level), "up"));
            String how = "in the closing auction only at or above " + g(lim) + " (LOC, close " + closes + " of " + last + ")";
            Broker.Fate f = ib.fate(ib.place(c, Broker.NewOrder.of("SELL", "LOC", qty, lim, 0, ref + "-close", "DAY")), 20);
            List<String> msgs = f.messages();
            String fate = f.fate();
            if (fate.equals("dead") && msgs.stream().anyMatch(m -> m.startsWith("387") || m.contains("Error 387"))) {
                Broker.Fate f2 = ib.fate(ib.place(c, Broker.NewOrder.of("SELL", "LMT", qty, lim, 0, ref + "-close", "DAY")), 20);
                how += " refused (IBKR " + msgs.get(0).split(":")[0] + ": no LOC here); a limit at " + g(lim) + " for the rest of the day instead";
                fate = f2.fate();
                msgs = f2.messages();
            }
            lines.add(sym + ": sell " + g(qty) + " " + how + ": " + fate + said(msgs));
        }
    }
}
