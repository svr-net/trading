package net.svrnet.bridge;

import java.util.List;

/**
 * What the order steps need from IBKR, as plain records: the only class that talks to IBKR's API implements it
 * (Ib); the checks use a stand-in (tools/ibkr/java tests). Prices are IBKR's (LSE shares in its own scale).
 */
interface Broker {
    /** A share at IBKR: its id, symbol, exchange (routing), primary exchange and currency. */
    record Share(int conId, String symbol, String exchange, String primaryExchange, String currency) {
        Share routed(String ex) {
            return new Share(conId, symbol, ex, primaryExchange, currency);
        }
    }

    /** A holding: its share, shares held, average cost. */
    record Position(Share share, double qty, double avgCost) { }

    /** An order: what it is and where it stands (an id, IBKR's permanent id, its status and fills). */
    record Ord(int orderId, long permId, int parentId, Share share, String action, String type, double qty, double limit,
               double aux, double trailPercent, double trailStop, String ref, String tif, String goodTill, String status,
               double filled, double remaining) {
        boolean dead() {
            return status.equals("Cancelled") || status.equals("ApiCancelled") || status.equals("Inactive");
        }
    }

    /** A fill: its share, side (BOT or SLD), shares, price, the order's reference and permanent id, its id, the time, its commission. */
    record Fill(Share share, String side, double shares, double price, String ref, long permId, String execId, String time,
                double commission, double realizedPnl) { }

    /** The share's price increments by price band (low edge, increment), its smallest tick and price magnifier. */
    record Ticks(double[][] ladder, double minTick, double magnifier) {
        double at(double price) {
            double inc = minTick;
            if (ladder != null) for (double[] r : ladder) if (price >= r[0] - 1e-12) inc = r[1];
            return inc > 0 ? inc : 0.01;
        }
    }

    /** A holding in the account's portfolio: its share, shares, IBKR's price, unrealised profit and average cost. */
    record Holding(Share share, double qty, double marketPrice, double unrealizedPnl, double avgCost) { }

    /** A new order to place (the fields an order step sets; zero or null: not set). */
    record NewOrder(String action, String type, double qty, double limit, double aux, String ref, String tif, String goodTill,
                    int parentId, boolean transmit, double trailPercent, double trailStop, double limitOffset) {
        static NewOrder of(String action, String type, double qty, double limit, double aux, String ref, String tif) {
            return new NewOrder(action, type, qty, limit, aux, ref, tif, null, 0, true, 0, 0, 0);
        }

        NewOrder child(int parent, boolean transmitNow) {
            return new NewOrder(action, type, qty, limit, aux, ref, tif, goodTill, parent, transmitNow, trailPercent, trailStop, limitOffset);
        }

        NewOrder held() {  // not sent yet: its child sends both
            return new NewOrder(action, type, qty, limit, aux, ref, tif, goodTill, parentId, false, trailPercent, trailStop, limitOffset);
        }

        NewOrder until(String goodTillUtc) {
            return new NewOrder(action, type, qty, limit, aux, ref, tif, goodTillUtc, parentId, transmit, trailPercent, trailStop, limitOffset);
        }
    }

    /** IBKR's answer to an order after a wait: live (working), filled or dead, with its messages (numbers masked). */
    record Fate(String fate, List<String> messages) { }

    /** The share, qualified at IBKR (null when IBKR does not know it). */
    Share qualify(Share s);

    /** The share's sale contract (its id, routed to its primary exchange) and its price increments; null when unknown. */
    Share exitShare(Share s);

    Ticks ticks(Share s);

    /** IBKR's last close (its daily bars, else the delayed quote's close or last); NaN when neither answers. */
    double referenceClose(Share s);

    /** The lowest low of the last 20 days (else the delayed quote's lowest field); NaN when neither answers. */
    double floorPrice(Share s);

    List<Position> positions();

    /** The account's portfolio (IBKR's prices and unrealised profit). */
    List<Holding> portfolio();

    /** The open orders of every client of the account. */
    List<Ord> openOrders();

    /** The completed orders of the day (empty when IBKR does not answer in time). */
    List<Ord> completedOrders();

    /** The day's fills of the account. */
    List<Fill> fills();

    /** Places the order (its id set here) and returns its id. */
    int place(Share s, NewOrder o);

    void cancel(int orderId);

    /** Where the order stands after up to `seconds` (order_fate). */
    Fate fate(int orderId, int seconds);

    void sleep(double seconds);
}
