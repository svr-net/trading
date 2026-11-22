package net.svrnet.bridge;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * IBC's command server (CommandServerPort, set to 7462 and bound to 127.0.0.1 in the container's IBC settings):
 * one command per connection, IBC's own answer back (e.g. "OK" or "ERROR ..."). RESTART restarts IB Gateway
 * and logs in again; RECONNECTACCOUNT and RECONNECTDATA reconnect its sessions without a restart.
 */
final class Ibc {
    static String send(String command) {
        int port = Integer.parseInt(Bridge.env("IBC_COMMAND_PORT", "7462"));
        try (Socket s = new Socket()) {
            s.connect(new InetSocketAddress("127.0.0.1", port), 5000);
            s.setSoTimeout(30_000);
            OutputStream o = s.getOutputStream();
            o.write((command + "\n").getBytes(StandardCharsets.US_ASCII));
            o.flush();
            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.US_ASCII));
            StringBuilder answer = new StringBuilder();
            String line;
            while ((line = in.readLine()) != null) {  // IBC answers, then closes (or the timeout ends it)
                line = line.strip();
                if (line.isEmpty()) continue;
                answer.append(answer.length() == 0 ? "" : " ").append(line);
                if (line.startsWith("OK") || line.startsWith("ERROR")) break;
            }
            return answer.length() == 0 ? "ERROR no answer" : answer.toString();
        } catch (IOException e) {
            return "ERROR IBC's command server does not answer on 127.0.0.1:" + port + " (" + e.getMessage() + ")";
        }
    }

    private Ibc() {
    }
}
