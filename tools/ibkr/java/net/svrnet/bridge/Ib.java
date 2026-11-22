package net.svrnet.bridge;

import com.ib.client.Bar;
import com.ib.client.CommissionAndFeesReport;
import com.ib.client.Contract;
import com.ib.client.ContractDetails;
import com.ib.client.Decimal;
import com.ib.client.DefaultEWrapper;
import com.ib.client.EClientSocket;
import com.ib.client.EJavaSignal;
import com.ib.client.EReader;
import com.ib.client.Execution;
import com.ib.client.ExecutionFilter;
import com.ib.client.Order;
import com.ib.client.OrderState;
import com.ib.client.PriceIncrement;
import com.ib.client.TickAttrib;

import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The one class that talks to IBKR: IBKR's own Java client (com.ib.client) behind the Broker interface, every
 * request answered synchronously (the order steps are sequential, as submit.py's with ib_async were). Each step
 * connects with its own client id (submit.py's: the market's 17.. range), retrying a client id still held, and
 * every request answers within a minute or fails (a gateway that no longer reaches IBKR's servers made steps
 * hang for hours, 6 Nov).
 */
final class Ib extends DefaultEWrapper implements Broker, AutoCloseable {
    static final int REQUEST_TIMEOUT = 60;
    static final List<String> DEAD = List.of("Cancelled", "ApiCancelled", "Inactive");

    /** A request's answer: what came back, ended (or failed) when the latch opens. */
    static final class Req {
        final List<Object> items = new ArrayList<>();
        final CountDownLatch done = new CountDownLatch(1);
        final List<String> errors = new ArrayList<>();
    }

    /** An order this connection placed, or saw: its last status and IBKR's messages. */
    static final class Live {
        volatile String status = "";
        volatile double filled;
        volatile double remaining;
        volatile long permId;
        final List<String> log = new ArrayList<>();
    }

    private final EJavaSignal signal = new EJavaSignal();
    private final EClientSocket client = new EClientSocket(this, signal);
    private final AtomicInteger reqIds = new AtomicInteger(1000);
    private final Map<Integer, Req> reqs = new ConcurrentHashMap<>();
    private final Map<Integer, Live> orders = new ConcurrentHashMap<>();
    private final Map<String, double[]> commissions = new ConcurrentHashMap<>();  // execId: commission, realised profit
    private final Map<Integer, Map<Integer, Double>> quotes = new ConcurrentHashMap<>();  // a stream's fields by IBKR's tick type
    private final Map<Integer, Integer> dataTypes = new ConcurrentHashMap<>();  // a stream's market data type (1 real time, 3 delayed, ...)
    private volatile CountDownLatch timeReq;
    private volatile long serverTime;
    private volatile List<String> accounts = List.of();
    private final CountDownLatch ready = new CountDownLatch(1);
    private final List<String> notes = new ArrayList<>();
    private volatile Req openReq, positionsReq, completedReq, portfolioReq;
    private volatile Req ruleReq;
    private volatile int ruleId = -1;
    private volatile int nextId = -1;
    volatile String barsError = "";  // the last bars request's IBKR error (code: message), for the bars fetch's summary
    private volatile String account = "";

    /** Connects, retrying: a client id still held (a process ending, a gateway restarting) is often free again within a minute. */
    static Ib connect(String host, int port, int clientId, int tries) {
        RuntimeException last = null;
        for (int k = 0; k < tries; k++) {
            Ib ib = new Ib();
            try {
                ib.open(host, port, clientId);
                return ib;
            } catch (RuntimeException e) {
                ib.close();
                last = e;
                if (k == tries - 1) break;
                System.out.println("  connecting as client " + clientId + ": " + e.getMessage() + "; again in 20 seconds");
                ib.sleep(20);
            }
        }
        throw last;
    }

    static Ib connect(int clientId) {
        return connect(Bridge.env("IB_HOST", "127.0.0.1"), Integer.parseInt(Bridge.env("IB_PORT", "4002")), clientId, 3);
    }

    private void open(String host, int port, int clientId) {
        client.eConnect(host, port, clientId);
        if (!client.isConnected()) throw new IllegalStateException("IB Gateway did not accept the API connection at " + host + ":" + port);
        EReader reader = new EReader(client, signal);
        reader.start();
        Thread pump = new Thread(() -> {
            while (client.isConnected()) {
                signal.waitForSignal();
                try {
                    reader.processMsgs();
                } catch (Exception e) {
                    note("reader: " + e.getMessage());
                }
            }
        }, "ibkr-" + clientId);
        pump.setDaemon(true);
        pump.start();
        if (!await(ready, 20)) throw new IllegalStateException("IB Gateway connected but sent no order id within 20 seconds");
    }

    boolean connected() {
        return client.isConnected();
    }

    String account() {
        return account;
    }

    @Override
    public void close() {
        if (client.isConnected()) client.eDisconnect();
        signal.issueSignal();  // the reader's thread wakes, sees the connection closed and ends
    }

    // --- requests
    private static boolean await(CountDownLatch l, int seconds) {
        try {
            return l.await(seconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private Req req(int id) {
        Req r = new Req();
        reqs.put(id, r);
        return r;
    }

    private List<Object> answer(int id, Req r, int seconds, String what) {
        boolean ok = await(r.done, seconds);
        reqs.remove(id);
        if (!ok) throw new IllegalStateException(what + ": no answer from IBKR in " + seconds + " s");
        synchronized (r) {
            return new ArrayList<>(r.items);
        }
    }

    static Contract contract(Share s) {
        Contract c = new Contract();
        if (s.conId() != 0) c.conid(s.conId());
        if (s.symbol() != null && !s.symbol().isEmpty() && s.conId() == 0) c.symbol(s.symbol());
        c.secType("STK");
        if (s.exchange() != null && !s.exchange().isEmpty()) c.exchange(s.exchange());
        if (s.currency() != null && !s.currency().isEmpty() && s.conId() == 0) c.currency(s.currency());
        if (s.primaryExchange() != null && !s.primaryExchange().isEmpty() && s.conId() == 0) c.primaryExch(s.primaryExchange());
        return c;
    }

    static Share share(Contract c) {
        return new Share(c.conid(), c.symbol(), c.exchange(), c.primaryExch(), c.currency());
    }

    List<ContractDetails> details(Share s) {
        return details(s, REQUEST_TIMEOUT);
    }

    List<ContractDetails> details(Share s, int seconds) {
        int id = reqIds.incrementAndGet();
        Req r = req(id);
        client.reqContractDetails(id, contract(s));
        List<ContractDetails> out = new ArrayList<>();
        for (Object o : answer(id, r, seconds, "contract details")) out.add((ContractDetails) o);
        return out;
    }

    /** Whether IBKR knows the share (one contract, as ib_async's qualifyContracts: an ambiguous one is not an answer). */
    @Override
    public Share qualify(Share s) {
        try {
            List<ContractDetails> d = details(s);
            Map<Integer, Contract> byId = new LinkedHashMap<>();
            for (ContractDetails x : d) byId.putIfAbsent(x.contract().conid(), x.contract());
            if (byId.size() != 1) return null;
            Contract c = byId.values().iterator().next();
            Share q = share(c);
            return q.conId() == 0 ? null : q;
        } catch (RuntimeException e) {
            return null;
        }
    }

    @Override
    public Share exitShare(Share s) {
        Share q = qualify(new Share(s.conId(), s.conId() != 0 ? "" : s.symbol(), "", "", s.conId() != 0 ? "" : s.currency()));
        if (q == null) return null;
        String ex = q.primaryExchange() != null && !q.primaryExchange().isEmpty() ? q.primaryExchange() : q.exchange();
        return q.routed(ex);
    }

    /** The share's price increments (IBKR's market rule for its exchange; the LSE's depend on the price), smallest tick, magnifier. */
    @Override
    public Ticks ticks(Share s) {
        if (s == null) return null;
        ContractDetails d;
        try {
            List<ContractDetails> all = details(s);
            if (all.isEmpty()) return null;
            d = all.get(0);
        } catch (RuntimeException e) {
            return null;
        }
        String[] ids = d.marketRuleIds() == null ? new String[0] : Arrays.stream(d.marketRuleIds().split(",")).filter(x -> !x.isEmpty()).toArray(String[]::new);
        List<String> exs = Arrays.asList((d.validExchanges() == null ? "" : d.validExchanges()).split(","));
        String rid = s.exchange() != null && exs.contains(s.exchange()) && ids.length == exs.size() ? ids[exs.indexOf(s.exchange())] : ids.length > 0 ? ids[0] : null;
        double[][] ladder = null;
        if (rid != null) {
            try {
                PriceIncrement[] incs = marketRule(Integer.parseInt(rid.trim()));
                if (incs != null && incs.length > 0) {
                    ladder = new double[incs.length][];
                    for (int i = 0; i < incs.length; i++) ladder[i] = new double[] {incs[i].lowEdge(), incs[i].increment()};
                    Arrays.sort(ladder, Comparator.comparingDouble(r -> r[0]));
                }
            } catch (RuntimeException e) {
                ladder = null;
            }
        }
        double mag = d.priceMagnifier() > 0 ? d.priceMagnifier() : 1.0;
        return new Ticks(ladder, d.minTick(), mag);
    }

    private synchronized PriceIncrement[] marketRule(int id) {
        Req r = new Req();
        ruleId = id;
        ruleReq = r;
        client.reqMarketRule(id);
        boolean ok = await(r.done, 10);
        ruleReq = null;
        if (!ok || r.items.isEmpty()) return null;
        return (PriceIncrement[]) r.items.get(0);
    }

    /** The share's daily bars (TRADES, regular hours), or an empty list when IBKR does not answer (no data subscription: 162). */
    List<Bar> bars(Share s, String duration, int seconds) {
        int id = reqIds.incrementAndGet();
        Req r = req(id);
        client.reqHistoricalData(id, contract(s), "", duration, "1 day", "TRADES", 1, 1, false, null);
        try {
            List<Bar> out = new ArrayList<>();
            for (Object o : answer(id, r, seconds, "daily bars")) out.add((Bar) o);
            synchronized (r) {
                barsError = r.errors.isEmpty() ? (out.isEmpty() ? "no error: no bars" : "") : r.errors.get(r.errors.size() - 1);
            }
            return out;
        } catch (RuntimeException e) {
            client.cancelHistoricalData(id);
            barsError = "exception: " + e.getMessage();
            return List.of();
        }
    }

    /** The free delayed quote (frozen when the market is shut): its fields by name, after up to `seconds`, once any of `want` has come. */
    Map<String, Double> quote(Share s, double seconds, String... want) {
        int id = stream(s, "", 4);
        try {
            for (int i = 0; i < seconds * 2; i++) {
                sleep(0.5);
                Map<String, Double> named = fields(id);
                boolean any = false;
                for (String w : want) any |= named.containsKey(w);
                if (any) {
                    sleep(1.0);  // the rest of the quote's fields
                    return fields(id);
                }
            }
            return fields(id);
        } finally {
            unstream(id);
        }
    }

    /** Streams the share's market data (`generic`: IBKR's generic ticks, 225 the auction's; `type`: 1 real time, 3 delayed where not subscribed, 4 delayed frozen). */
    int stream(Share s, String generic, int type) {
        int id = reqIds.incrementAndGet();
        quotes.put(id, new ConcurrentHashMap<>());
        client.reqMarketDataType(type);
        client.reqMktData(id, contract(s), generic, false, false, null);
        return id;
    }

    void unstream(int id) {
        client.cancelMktData(id);
        quotes.remove(id);
        dataTypes.remove(id);
    }

    /** A stream's latest fields by name (bid, ask, last, high, low, close, open, volume, bidSize, askSize, auctionPrice, auctionVolume, auctionImbalance). */
    Map<String, Double> fields(int id) {
        Map<Integer, Double> q = quotes.getOrDefault(id, Map.of());
        Map<String, Double> m = new LinkedHashMap<>();
        for (Map.Entry<Integer, Double> e : q.entrySet()) {
            String n = switch (e.getKey()) {
                case 0, 69 -> "bidSize";
                case 1, 66 -> "bid";
                case 2, 67 -> "ask";
                case 3, 70 -> "askSize";
                case 4, 68 -> "last";
                case 6, 72 -> "high";
                case 7, 73 -> "low";
                case 8, 74 -> "volume";
                case 9, 75 -> "close";
                case 14, 76 -> "open";
                case 34 -> "auctionVolume";
                case 35 -> "auctionPrice";
                case 36 -> "auctionImbalance";
                default -> null;
            };
            double v = e.getValue();
            boolean price = n != null && !n.endsWith("Size") && !n.equals("volume") && !n.startsWith("auctionV") && !n.equals("auctionImbalance");
            if (n != null && Double.isFinite(v) && Math.abs(v) < 1e300 && (price ? v > 0 : v > -1e300)) m.put(n, v);
        }
        return m;
    }

    /** A stream's market data type as IBKR said it (1 real time, 2 frozen, 3 delayed, 4 delayed frozen; 0 not said yet). */
    int dataType(int id) {
        return dataTypes.getOrDefault(id, 0);
    }

    /** IBKR's clock (its servers' time, seconds since 1970), or 0 when it does not answer within `seconds`. */
    long serverTime(int seconds) {
        CountDownLatch l = new CountDownLatch(1);
        timeReq = l;
        client.reqCurrentTime();
        return await(l, seconds) ? serverTime : 0;
    }

    List<String> accounts() {
        return accounts;
    }

    @Override
    public double referenceClose(Share s) {
        List<Bar> b = bars(s, "2 D", 20);
        if (!b.isEmpty() && b.get(b.size() - 1).close() > 0) return b.get(b.size() - 1).close();
        Map<String, Double> q = quote(s, 8, "close", "last");
        if (q.containsKey("close")) return q.get("close");
        if (q.containsKey("last")) return q.get("last");
        return Double.NaN;
    }

    @Override
    public double floorPrice(Share s) {
        List<Double> lows = new ArrayList<>();
        List<Bar> b = bars(s, "1 M", 20);
        for (Bar x : b.subList(Math.max(0, b.size() - 20), b.size())) if (x.low() > 0) lows.add(x.low());
        if (lows.isEmpty()) {
            Map<String, Double> q = quote(s, 10, "low", "last", "bid", "close");
            for (String k : List.of("low", "last", "bid", "close")) if (q.containsKey(k)) lows.add(q.get(k));
        }
        if (lows.isEmpty()) return Double.NaN;
        double lo = lows.stream().min(Double::compare).get();
        Ticks t = ticks(s);
        return Holds.roundTo(lo, t == null ? 0.01 : t.at(lo), "down");
    }

    @Override
    public synchronized List<Position> positions() {
        Req r = new Req();
        positionsReq = r;
        client.reqPositions();
        boolean ok = await(r.done, REQUEST_TIMEOUT);
        client.cancelPositions();
        positionsReq = null;
        if (!ok) throw new IllegalStateException("positions: no answer from IBKR in " + REQUEST_TIMEOUT + " s");
        List<Position> out = new ArrayList<>();
        synchronized (r) {
            for (Object o : r.items) out.add((Position) o);
        }
        return out;
    }

    @Override
    public synchronized List<Holding> portfolio() {
        Req r = new Req();
        portfolioReq = r;
        client.reqAccountUpdates(true, account);
        boolean ok = await(r.done, REQUEST_TIMEOUT);
        client.reqAccountUpdates(false, account);
        portfolioReq = null;
        if (!ok) throw new IllegalStateException("portfolio: no answer from IBKR in " + REQUEST_TIMEOUT + " s");
        Map<Integer, Holding> out = new LinkedHashMap<>();
        synchronized (r) {
            for (Object o : r.items) out.put(((Holding) o).share().conId(), (Holding) o);
        }
        return new ArrayList<>(out.values());
    }

    @Override
    public synchronized List<Ord> openOrders() {
        Req r = new Req();
        openReq = r;
        client.reqAllOpenOrders();
        boolean ok = await(r.done, REQUEST_TIMEOUT);
        openReq = null;
        if (!ok) throw new IllegalStateException("open orders: no answer from IBKR in " + REQUEST_TIMEOUT + " s");
        Map<Long, Ord> out = new LinkedHashMap<>();
        synchronized (r) {
            for (Object o : r.items) {
                Ord x = withStatus((Ord) o);
                out.put(x.permId() != 0 ? x.permId() : -x.orderId(), x);
            }
        }
        return new ArrayList<>(out.values());
    }

    private Ord withStatus(Ord o) {
        Live l = orders.get(o.orderId());
        if (l == null || l.status.isEmpty() || (o.permId() != 0 && l.permId != 0 && l.permId != o.permId())) return o;
        return new Ord(o.orderId(), o.permId(), o.parentId(), o.share(), o.action(), o.type(), o.qty(), o.limit(), o.aux(), o.trailPercent(),
                o.trailStop(), o.ref(), o.tif(), o.goodTill(), l.status, l.filled, l.remaining);
    }

    /** IBKR's completed orders, or none when it does not answer within 20 seconds (IB Gateway at times never ends the list, 11 Nov). */
    @Override
    public synchronized List<Ord> completedOrders() {
        Req r = new Req();
        completedReq = r;
        client.reqCompletedOrders(false);
        boolean ok = await(r.done, 20);
        completedReq = null;
        if (!ok) {
            System.err.println("  completed orders: no answer from IBKR in 20 s; without them");
            return List.of();
        }
        List<Ord> out = new ArrayList<>();
        synchronized (r) {
            for (Object o : r.items) out.add((Ord) o);
        }
        return out;
    }

    @Override
    public List<Fill> fills() {
        int id = reqIds.incrementAndGet();
        Req r = req(id);
        client.reqExecutions(id, new ExecutionFilter());
        List<Object> got = answer(id, r, REQUEST_TIMEOUT, "fills");
        sleep(0.5);  // their commission reports follow the list
        List<Fill> out = new ArrayList<>();
        for (Object o : got) {
            Object[] x = (Object[]) o;
            Contract c = (Contract) x[0];
            Execution e = (Execution) x[1];
            double[] cr = commissions.getOrDefault(e.execId(), new double[] {0, 0});
            out.add(new Fill(share(c), e.side(), num(e.shares()), e.price(), e.orderRef() == null ? "" : e.orderRef(), permIdOf(e), e.execId(),
                    hhmmUtc(e.time()), cr[0], cr[1]));
        }
        return out;
    }

    /** An execution's time (IBKR's "yyyyMMdd HH:mm:ss zone", or "yyyyMMdd-HH:mm:ss" in UTC) as HH:mm UTC, as submit.py's report. */
    static String hhmmUtc(String t) {
        if (t == null || t.isBlank()) return "";
        try {
            String[] p = t.trim().split("\\s+");
            if (p.length >= 3) {
                LocalDateTime l = LocalDateTime.parse(p[0] + " " + p[1], DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss"));
                return ZonedDateTime.of(l, ZoneId.of(p[2])).withZoneSameInstant(ZoneOffset.UTC).format(DateTimeFormatter.ofPattern("HH:mm"));
            }
            String s = p[0].contains("-") ? p[0] : p[0] + "-" + (p.length > 1 ? p[1] : "00:00:00");
            LocalDateTime l = LocalDateTime.parse(s, DateTimeFormatter.ofPattern("yyyyMMdd-HH:mm:ss"));
            return l.format(DateTimeFormatter.ofPattern("HH:mm"));
        } catch (RuntimeException e) {
            return "";
        }
    }

    @Override
    public synchronized int place(Share s, NewOrder n) {
        int id = nextId++;
        Order o = new Order();
        o.action(n.action());
        o.orderType(n.type());
        o.totalQuantity(Decimal.get(n.qty()));
        if (n.limit() != 0) o.lmtPrice(n.limit());
        if (n.aux() != 0) o.auxPrice(n.aux());
        if (n.trailPercent() != 0) o.trailingPercent(n.trailPercent());
        if (n.trailStop() != 0) o.trailStopPrice(n.trailStop());
        if (n.limitOffset() != 0) o.lmtPriceOffset(n.limitOffset());
        if (n.ref() != null) o.orderRef(n.ref());
        if (n.tif() != null) o.tif(n.tif());
        if (n.goodTill() != null) o.goodTillDate(n.goodTill());
        if (n.parentId() != 0) o.parentId(n.parentId());
        o.transmit(n.transmit());
        orders.put(id, new Live());
        client.placeOrder(id, contract(s), o);
        return id;
    }

    /** cancelOrder's arguments differ between API versions (a String in older ones, an OrderCancel in newer). */
    @Override
    public void cancel(int orderId) {
        for (Method m : client.getClass().getMethods()) {
            if (!m.getName().equals("cancelOrder") || m.getParameterCount() != 2) continue;
            try {
                Class<?> t = m.getParameterTypes()[1];
                Object arg = t == String.class ? "" : t.getDeclaredConstructor().newInstance();
                m.invoke(client, orderId, arg);
                return;
            } catch (ReflectiveOperationException e) {
                note("cancel: " + e.getMessage());
            }
        }
    }

    /**
     * Where an order stands after up to `seconds`: live (Submitted or PreSubmitted), filled, or dead (cancelled,
     * inactive, refused). Without a clear status, whether IBKR lists it among its open orders, else its fills.
     */
    @Override
    public Fate fate(int orderId, int seconds) {
        Live l = orders.computeIfAbsent(orderId, k -> new Live());
        for (int i = 0; i < seconds * 2; i++) {
            sleep(0.5);
            String st = l.status;
            if (st.equals("Filled") || DEAD.contains(st) || st.equals("Submitted") || st.equals("PreSubmitted")) break;
        }
        List<String> msgs = new ArrayList<>();
        synchronized (l.log) {
            for (String m : l.log) {
                String x = m.replaceAll("\\d+\\.\\d+", "#");
                x = x.length() > 160 ? x.substring(0, 160) : x;
                if (!msgs.contains(x)) msgs.add(x);
            }
        }
        String st = l.status;
        if (st.equals("Filled")) return new Fate("filled", msgs);
        if (DEAD.contains(st)) return new Fate("dead", msgs);
        if (st.equals("Submitted") || st.equals("PreSubmitted")) return new Fate("live", msgs);
        try {
            for (Ord o : openOrders()) if (!o.dead() && (o.orderId() == orderId || (l.permId != 0 && o.permId() == l.permId))) return new Fate("live", msgs);
            for (Fill f : fills()) if (l.permId != 0 && f.permId() == l.permId) return new Fate("filled", msgs);
        } catch (RuntimeException e) {
            msgs.add(e.getMessage());
        }
        return new Fate("dead", msgs);
    }

    @Override
    public void sleep(double seconds) {
        try {
            Thread.sleep((long) (seconds * 1000));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    synchronized List<String> notes() {
        return new ArrayList<>(notes);
    }

    private synchronized void note(String m) {
        if (!notes.contains(m)) notes.add(m);
    }

    static double num(Decimal d) {
        if (d == null) return 0;
        try {
            double v = Double.parseDouble(d.toString());
            return Double.isFinite(v) ? v : 0;
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static long permIdOf(Object o) {
        try {
            Object v = o.getClass().getMethod("permId").invoke(o);
            return ((Number) v).longValue();
        } catch (ReflectiveOperationException e) {
            return 0;
        }
    }

    private static String str(Object o, String getter) {
        try {
            Object v = o.getClass().getMethod(getter).invoke(o);
            return v == null ? "" : v.toString();
        } catch (ReflectiveOperationException e) {
            return "";
        }
    }

    static Ord ord(Contract c, Order o, OrderState st) {
        String status = st == null ? "" : str(st, "getStatus");
        return new Ord(o.orderId(), permIdOf(o), o.parentId(), share(c), str(o, "getAction"), str(o, "getOrderType"), num(o.totalQuantity()),
                fin(o.lmtPrice()), fin(o.auxPrice()), fin(o.trailingPercent()), fin(o.trailStopPrice()), o.orderRef() == null ? "" : o.orderRef(),
                str(o, "getTif"), o.goodTillDate(), status, 0, num(o.totalQuantity()));
    }

    private static double fin(double x) {
        return Double.isFinite(x) && Math.abs(x) < 1e300 ? x : 0;
    }

    // --- EWrapper
    @Override
    public void nextValidId(int id) {
        if (nextId < id) nextId = id;
        ready.countDown();
    }

    @Override
    public void managedAccounts(String list) {
        if (list == null || list.isBlank()) return;
        List<String> a = new ArrayList<>();
        for (String x : list.split(",")) if (!x.isBlank()) a.add(x.trim());
        accounts = a;
        if (!a.isEmpty()) account = a.get(0);
    }

    @Override
    public void currentTime(long time) {
        serverTime = time;
        CountDownLatch l = timeReq;
        if (l != null) l.countDown();
    }

    @Override
    public void tickSize(int tickerId, int field, Decimal size) {
        Map<Integer, Double> q = quotes.get(tickerId);
        if (q != null && size != null && size.isValid()) q.put(field, num(size));
    }

    @Override
    public void marketDataType(int reqId, int type) {
        if (quotes.containsKey(reqId)) dataTypes.put(reqId, type);
    }

    @Override
    public void contractDetails(int reqId, ContractDetails d) {
        Req r = reqs.get(reqId);
        if (r != null) synchronized (r) {
            r.items.add(d);
        }
    }

    @Override
    public void contractDetailsEnd(int reqId) {
        Req r = reqs.get(reqId);
        if (r != null) r.done.countDown();
    }

    @Override
    public void historicalData(int reqId, Bar bar) {
        Req r = reqs.get(reqId);
        if (r != null) synchronized (r) {
            r.items.add(bar);
        }
    }

    @Override
    public void historicalDataEnd(int reqId, String start, String end) {
        Req r = reqs.get(reqId);
        if (r != null) r.done.countDown();
    }

    @Override
    public void marketRule(int id, PriceIncrement[] increments) {
        Req r = ruleReq;
        if (r != null && id == ruleId) {
            synchronized (r) {
                r.items.add(increments);
            }
            r.done.countDown();
        }
    }

    @Override
    public void tickPrice(int tickerId, int field, double price, TickAttrib attrib) {
        Map<Integer, Double> q = quotes.get(tickerId);
        if (q != null && (price > 0 || field == 36)) q.put(field, price);
    }

    @Override
    public void position(String acct, Contract c, Decimal pos, double avgCost) {
        Req r = positionsReq;
        if (r != null) synchronized (r) {
            r.items.add(new Position(share(c), num(pos), avgCost));
        }
    }

    @Override
    public void positionEnd() {
        Req r = positionsReq;
        if (r != null) r.done.countDown();
    }

    @Override
    public void updatePortfolio(Contract c, Decimal position, double marketPrice, double marketValue, double averageCost,
                                double unrealizedPNL, double realizedPNL, String accountName) {
        Req r = portfolioReq;
        if (r != null) synchronized (r) {
            r.items.add(new Holding(share(c), num(position), marketPrice, unrealizedPNL, averageCost));
        }
    }

    @Override
    public void accountDownloadEnd(String accountName) {
        Req r = portfolioReq;
        if (r != null) r.done.countDown();
    }

    @Override
    public void openOrder(int orderId, Contract c, Order o, OrderState st) {
        Live l = orders.get(orderId);
        if (l != null && l.permId == 0) l.permId = permIdOf(o);
        Req r = openReq;
        if (r != null) synchronized (r) {
            r.items.add(ord(c, o, st));
        }
    }

    @Override
    public void openOrderEnd() {
        Req r = openReq;
        if (r != null) r.done.countDown();
    }

    @Override
    public void completedOrder(Contract c, Order o, OrderState st) {
        Req r = completedReq;
        if (r != null) synchronized (r) {
            r.items.add(ord(c, o, st));
        }
    }

    @Override
    public void completedOrdersEnd() {
        Req r = completedReq;
        if (r != null) r.done.countDown();
    }

    @Override
    public void execDetails(int reqId, Contract c, Execution e) {
        Req r = reqs.get(reqId);
        if (r != null) synchronized (r) {
            r.items.add(new Object[] {c, e});
        }
    }

    @Override
    public void execDetailsEnd(int reqId) {
        Req r = reqs.get(reqId);
        if (r != null) r.done.countDown();
    }

    @Override
    public void commissionAndFeesReport(CommissionAndFeesReport r) {
        if (r != null && r.execId() != null) commissions.put(r.execId(), new double[] {r.commissionAndFees(), r.realizedPNL()});
    }

    // orderStatus: permId an int in older API versions, a long in newer ones (both declared; the other is unused)
    public void orderStatus(int id, String st, Decimal filled, Decimal remaining, double avgFillPrice, int permId, int parentId,
                            double lastFillPrice, int clientId, String whyHeld, double mktCapPrice) {
        status(id, st, filled, remaining, permId);
    }

    public void orderStatus(int id, String st, Decimal filled, Decimal remaining, double avgFillPrice, long permId, int parentId,
                            double lastFillPrice, int clientId, String whyHeld, double mktCapPrice) {
        status(id, st, filled, remaining, permId);
    }

    private void status(int id, String st, Decimal filled, Decimal remaining, long permId) {
        Live l = orders.computeIfAbsent(id, k -> new Live());
        l.status = st == null ? "" : st;
        l.filled = num(filled);
        l.remaining = num(remaining);
        if (permId != 0) l.permId = permId;
    }

    // error: its arguments differ between API versions (all declared; the others are unused)
    public void error(int id, int code, String msg, String advancedOrderRejectJson) {
        onError(id, code, msg);
    }

    public void error(int id, long time, int code, String msg, String advancedOrderRejectJson) {
        onError(id, code, msg);
    }

    public void error(int id, int code, String msg) {
        onError(id, code, msg);
    }

    private void onError(int id, int code, String msg) {
        if (code == 2104 || code == 2106 || code == 2107 || code == 2108 || code == 2158 || code == 2119) return;  // data farm notices
        Live l = orders.get(id);
        if (l != null) {
            synchronized (l.log) {
                l.log.add(code + ": " + msg);
            }
            // refused (2139 paper trading not acknowledged, 201 rejected, 203 not allowed, 110 price, 103 duplicate id)
            if (code == 2139 || code == 201 || code == 203 || code == 110 || code == 103) l.status = "Inactive";
            return;
        }
        Req r = reqs.get(id);
        if (r != null && code != 2176 && code < 2100) {  // a request failed (162 no data, 200 no such contract): it ends now
            synchronized (r) {
                r.errors.add(code + ": " + msg);
            }
            r.done.countDown();
            return;
        }
        note(code + ": " + msg);  // the data's messages too (354 not subscribed, 10089, 10168, 101 too many lines): diag reports them
    }

    @Override
    public void error(String msg) {
        note(msg);
    }

    @Override
    public void error(Exception e) {
        note(String.valueOf(e.getMessage()));
    }
}
