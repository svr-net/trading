package net.svrnet.bridge;

import java.io.InputStream;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyStore;
import java.security.Principal;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLEngine;
import javax.net.ssl.X509ExtendedKeyManager;
import javax.net.ssl.X509KeyManager;

/**
 * The bridge's certificate (a PKCS#12 file the container's start script makes from DSM's PEM files). The key
 * manager reads the file again when it changes (checked at most every 10 minutes), so a renewed certificate
 * serves the next handshakes without a restart.
 */
final class Tls {
    private final Path file;
    private final char[] password;
    private volatile X509KeyManager current;
    private volatile long modified;
    private volatile long checked;

    Tls(Path file, char[] password) throws Exception {
        this.file = file;
        this.password = password;
        load();
    }

    SSLContext context() throws Exception {
        SSLContext ctx = SSLContext.getInstance("TLS");
        ctx.init(new X509ExtendedKeyManager[] {new Reloading()}, null, null);
        return ctx;
    }

    private void load() throws Exception {
        KeyStore ks = KeyStore.getInstance("PKCS12");
        try (InputStream in = Files.newInputStream(file)) {
            ks.load(in, password);
        }
        KeyManagerFactory kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
        kmf.init(ks, password);
        current = (X509KeyManager) kmf.getKeyManagers()[0];
        modified = Files.getLastModifiedTime(file).toMillis();
    }

    private X509KeyManager km() {
        long now = System.currentTimeMillis();
        if (now - checked > 600_000) {
            checked = now;
            try {
                if (Files.getLastModifiedTime(file).toMillis() != modified) {
                    load();
                    System.err.println("java bridge: certificate file changed: reloaded");
                }
            } catch (Exception e) {
                System.err.println("java bridge: certificate not reloaded: " + e.getMessage());
            }
        }
        return current;
    }

    private final class Reloading extends X509ExtendedKeyManager {
        public String[] getClientAliases(String t, Principal[] i) { return km().getClientAliases(t, i); }
        public String chooseClientAlias(String[] t, Principal[] i, Socket s) { return km().chooseClientAlias(t, i, s); }
        public String[] getServerAliases(String t, Principal[] i) { return km().getServerAliases(t, i); }
        public String chooseServerAlias(String t, Principal[] i, Socket s) { return km().chooseServerAlias(t, i, s); }
        public X509Certificate[] getCertificateChain(String a) { return km().getCertificateChain(a); }
        public PrivateKey getPrivateKey(String a) { return km().getPrivateKey(a); }

        @Override
        public String chooseEngineServerAlias(String t, Principal[] i, SSLEngine e) {
            X509KeyManager k = km();
            if (k instanceof X509ExtendedKeyManager) return ((X509ExtendedKeyManager) k).chooseEngineServerAlias(t, i, e);
            String[] a = k.getServerAliases(t, i);
            return a == null || a.length == 0 ? null : a[0];
        }
    }
}
