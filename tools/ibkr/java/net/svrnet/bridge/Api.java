package net.svrnet.bridge;

import java.lang.reflect.Field;

/** The version of IBKR's Java API built in (its highest server version), for /java and the start-up line. */
final class Api {
    static String version() {
        try {
            Class<?> c = Class.forName("com.ib.client.EClient");
            for (String name : new String[] {"MAX_VERSION", "MAX_SERVER_VERSION"}) {
                try {
                    Field f = c.getDeclaredField(name);
                    f.setAccessible(true);
                    return "server version " + f.get(null);
                } catch (NoSuchFieldException ignored) {
                    // the other name
                }
            }
            Class<?> s = Class.forName("com.ib.client.EClientSocket");
            Package p = s.getPackage();
            return p != null && p.getImplementationVersion() != null ? p.getImplementationVersion() : "built in";
        } catch (Exception e) {
            return "missing (" + e.getClass().getSimpleName() + ")";
        }
    }

    private Api() {
    }
}
