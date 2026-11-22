package net.svrnet.bridge;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** CSV as Python's csv module reads and writes it (quotes where needed, rows ended by CRLF). */
final class Csv {
    static List<String[]> read(Path p) throws IOException {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
            List<String> row = new ArrayList<>();
            StringBuilder cell = new StringBuilder();
            boolean quoted = false, any = false;
            int c;
            while ((c = r.read()) != -1) {
                char ch = (char) c;
                any = true;
                if (quoted) {
                    if (ch == '"') {
                        r.mark(1);
                        int n = r.read();
                        if (n == '"') cell.append('"');
                        else {
                            quoted = false;
                            if (n != -1) r.reset();
                        }
                    } else cell.append(ch);
                } else if (ch == '"') {
                    quoted = true;
                } else if (ch == ',') {
                    row.add(cell.toString());
                    cell.setLength(0);
                } else if (ch == '\n' || ch == '\r') {
                    if (ch == '\r') {
                        r.mark(1);
                        if (r.read() != '\n') r.reset();
                    }
                    row.add(cell.toString());
                    cell.setLength(0);
                    if (!(row.size() == 1 && row.get(0).isEmpty())) rows.add(row.toArray(new String[0]));
                    row = new ArrayList<>();
                    any = false;
                } else cell.append(ch);
            }
            if (any) {
                row.add(cell.toString());
                rows.add(row.toArray(new String[0]));
            }
        }
        return rows;
    }

    static void write(Path p, List<String[]> rows) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(p, StandardCharsets.UTF_8)) {
            for (String[] r : rows) {
                for (int i = 0; i < r.length; i++) {
                    if (i > 0) w.write(',');
                    w.write(Recordings.csv(r[i]));
                }
                w.write("\r\n");
            }
        }
    }

    private Csv() {
    }
}
