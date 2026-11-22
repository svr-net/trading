package net.svrnet.bridge;

import com.ib.client.Bar;
import com.ib.client.Contract;
import com.ib.client.ContractDetails;
import com.ib.client.Decimal;
import com.ib.client.DefaultEWrapper;
import com.ib.client.EClientSocket;
import com.ib.client.EJavaSignal;
import com.ib.client.EReader;
import com.ib.client.Order;
import com.ib.client.TickAttrib;

import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Does IBKR accept an order from this IB Gateway now? A limit buy of 1 share of the market's probe share at half
 * its last close (a day order that cannot fill), placed and cancelled at once (orderRef ofm-TEST, which no other
 * step touches). Its own API client id (the market's 17.. range + 5, as submit.py's helpers), so it never takes
 * another step's connection. The result as submit.py --test-order printed it: accepted, or refused with IBKR's
 * messages (2139: paper trading not acknowledged).
 */
final class TestOrder extends DefaultEWrapper {
    static final Map<String, String[]> PROBE = Map.of(
            "UK", new String[] {"VOD", "LSE", "GBP"}, "US", new String[] {"AAPL", "SMART", "USD"},
            "APAC", new String[] {"BHP", "ASX", "AUD"}, "EU", new String[] {"AIR", "SBF", "EUR"},
            "INDEX", new String[] {"EXS1", "IBIS", "EUR"});
    static final List<String> ORDER = List.of("UK", "EU", "US", "APAC", "INDEX");

    private final String market;
    private final EJavaSignal signal = new EJavaSignal();
    private final EClientSocket client = new EClientSocket(this, signal);
    private final List<String> messages = new ArrayList<>();
    private final CountDownLatch connected = new CountDownLatch(1);
    private final CountDownLatch detailsDone = new CountDownLatch(1);
    private final CountDownLatch historyDone = new CountDownLatch(1);
    private volatile int nextId = -1;
    private volatile ContractDetails details;
    private volatile double close = Double.NaN;
    private volatile double quote = Double.NaN;
    private volatile int orderId = -1;
    private volatile String status = "";
    private volatile boolean dead;

    TestOrder(String market) {
        this.market = market;
    }

    Map<String, Object> run() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("market", market);
        out.put("at", ZonedDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z")));
        String host = Bridge.env("IB_HOST", "127.0.0.1");
        int port = Integer.parseInt(Bridge.env("IB_PORT", "4002"));
        int clientId = 17 + 10 * ORDER.indexOf(market) + 5;
        client.eConnect(host, port, clientId);
        if (!client.isConnected()) {
            out.put("api", false);
            out.put("error", "IB Gateway did not accept the API connection at " + host + ":" + port);
            return out;
        }
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
        }, "ibkr-reader");
        pump.setDaemon(true);
        pump.start();
        try {
            if (!connected.await(15, TimeUnit.SECONDS)) {
                out.put("api", false);
                out.put("error", "IB Gateway connected but sent no order id within 15 seconds");
                return out;
            }
            out.put("api", true);
            String[] p = PROBE.get(market);
            out.put("probe", p[0]);
            Contract c = new Contract();
            c.symbol(p[0]);
            c.secType("STK");
            c.exchange(p[1]);
            c.currency(p[2]);
            client.reqContractDetails(1, c);
            detailsDone.await(15, TimeUnit.SECONDS);
            if (details == null) {
                out.put("error", p[0] + ": not found at IBKR");
                out.put("messages", masked());
                return out;
            }
            Contract contract = details.contract();
            client.reqHistoricalData(2, contract, "", "2 D", "1 day", "TRADES", 1, 1, false, null);
            historyDone.await(20, TimeUnit.SECONDS);
            if (!(close > 0)) {  // no daily bars (no data subscription): the free delayed quote
                client.reqMarketDataType(4);
                client.reqMktData(3, contract, "", false, false, null);
                for (int i = 0; i < 16 && !(quote > 0); i++) Thread.sleep(500);
                client.cancelMktData(3);
                close = quote;
            }
            if (!(close > 0)) {
                out.put("error", "no IBKR price for the probe share");
                out.put("messages", masked());
                return out;
            }
            double tick = details.minTick() > 0 ? details.minTick() : 0.01;
            double limit = BigDecimal.valueOf(0.5 * close).divide(BigDecimal.valueOf(tick), 0, RoundingMode.DOWN)
                    .multiply(BigDecimal.valueOf(tick)).doubleValue();
            Order o = new Order();
            o.action("BUY");
            o.orderType("LMT");
            o.totalQuantity(Decimal.get(1));
            o.lmtPrice(limit);
            o.tif("DAY");
            o.orderRef("ofm-TEST");
            o.transmit(true);
            orderId = nextId;
            client.placeOrder(orderId, contract, o);
            for (int i = 0; i < 30 && !dead && !isLive() && !status.equals("Filled"); i++) Thread.sleep(500);
            String fate = status.equals("Filled") ? "filled" : isLive() && !dead ? "live" : "dead";
            Boolean cancelled = null;
            if (fate.equals("live")) {
                cancel(orderId);
                for (int i = 0; i < 20 && !dead; i++) Thread.sleep(500);
                cancelled = dead;
            }
            List<String> msgs = masked();
            boolean refused = msgs.stream().anyMatch(m -> m.contains("2139") || m.toLowerCase().contains("paper-trading has not been acknowledged")
                    || m.toLowerCase().contains("paper trading has not been acknowledged"));
            out.put("fate", fate);
            out.put("accepted", fate.equals("live") || fate.equals("filled"));
            out.put("refused2139", refused);
            out.put("cancelled", cancelled);
            out.put("messages", msgs);
            return out;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            out.put("error", "interrupted");
            return out;
        } finally {
            client.eDisconnect();
        }
    }

    private boolean isLive() {
        return status.equals("Submitted") || status.equals("PreSubmitted");
    }

    private synchronized void note(String m) {
        if (!messages.contains(m)) messages.add(m);
    }

    private synchronized List<String> masked() {
        List<String> r = new ArrayList<>();
        for (String m : messages) {
            String s = m.replaceAll("\\d+\\.\\d+", "#");
            r.add(s.length() > 160 ? s.substring(0, 160) : s);
        }
        return r;
    }

    /** cancelOrder's arguments differ between API versions (a String in older ones, an OrderCancel in newer). */
    private void cancel(int id) {
        for (Method m : client.getClass().getMethods()) {
            if (!m.getName().equals("cancelOrder") || m.getParameterCount() != 2) continue;
            try {
                Class<?> t = m.getParameterTypes()[1];
                Object arg = t == String.class ? "" : t.getDeclaredConstructor().newInstance();
                m.invoke(client, id, arg);
                return;
            } catch (ReflectiveOperationException e) {
                note("cancel: " + e.getMessage());
            }
        }
    }

    // --- EWrapper: only what the test order needs (DefaultEWrapper ignores the rest)
    @Override
    public void nextValidId(int id) {
        nextId = id;
        connected.countDown();
    }

    @Override
    public void contractDetails(int reqId, ContractDetails d) {
        if (details == null) details = d;
    }

    @Override
    public void contractDetailsEnd(int reqId) {
        detailsDone.countDown();
    }

    @Override
    public void historicalData(int reqId, Bar bar) {
        if (bar.close() > 0) close = bar.close();
    }

    @Override
    public void historicalDataEnd(int reqId, String start, String end) {
        historyDone.countDown();
    }

    @Override
    public void tickPrice(int tickerId, int field, double price, TickAttrib attrib) {
        if (price > 0 && (field == 9 || field == 4 || field == 75 || field == 68)) quote = price;
    }

    // orderStatus: permId an int in older API versions, a long in newer ones (both declared; the other is unused)
    public void orderStatus(int id, String st, Decimal filled, Decimal remaining, double avgFillPrice, int permId, int parentId,
                            double lastFillPrice, int clientId, String whyHeld, double mktCapPrice) {
        status(id, st);
    }

    public void orderStatus(int id, String st, Decimal filled, Decimal remaining, double avgFillPrice, long permId, int parentId,
                            double lastFillPrice, int clientId, String whyHeld, double mktCapPrice) {
        status(id, st);
    }

    private void status(int id, String st) {
        if (id != orderId) return;
        status = st;
        if (st.equals("Cancelled") || st.equals("ApiCancelled") || st.equals("Inactive")) dead = true;
    }

    // error: its arguments differ between API versions (all declared; the others are unused)
    public void error(int id, int code, String msg, String advancedOrderRejectJson) {
        onError(id, code, msg);
    }

    public void error(int id, long time, int code, String msg, String advancedOrderRejectJson) {
        onError(id, code, msg);
    }

    public void error(int id, int code, String msg) {  // the oldest API versions
        onError(id, code, msg);
    }

    private void onError(int id, int code, String msg) {
        if (code == 2104 || code == 2106 || code == 2107 || code == 2108 || code == 2158 || code == 2119) return;  // data farm notices
        note(code + ": " + msg);
        // the order refused (2139 paper trading not acknowledged, 201 rejected, 203 not allowed, 110 price, 103 duplicate id)
        if (id == orderId && orderId >= 0 && (code == 2139 || code == 201 || code == 203 || code == 110 || code == 103)) dead = true;
        if (id == 1) detailsDone.countDown();
        if (id == 2) historyDone.countDown();
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
