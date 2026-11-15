// HTTP through libcurl for the loaders (EODData's API, the AIM list) and SMTP for the daily report.
#pragma once

#include <curl/curl.h>

#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <thread>

#include "eod/common.hpp"

namespace eod {

struct Response {
  long status = 0;
  std::string body, error;
};

inline std::size_t curlWrite(char* p, std::size_t size, std::size_t n, void* user) {
  static_cast<std::string*>(user)->append(p, size * n);
  return size * n;
}

inline void curlInit() {
  static const bool done = [] { return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK; }();
  (void)done;
}

inline std::string escape(const std::string& s) {
  curlInit();
  CURL* c = curl_easy_init();
  char* e = curl_easy_escape(c, s.c_str(), int(s.size()));
  std::string out = e ? e : "";
  curl_free(e);
  curl_easy_cleanup(c);
  return out;
}

// One GET. The proxy and CA settings come from the environment (HTTPS_PROXY, CURL_CA_BUNDLE).
inline Response get(const std::string& url, long timeout = 120) {
  curlInit();
  Response r;
  CURL* c = curl_easy_init();
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curlWrite);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout);
  curl_easy_setopt(c, CURLOPT_USERAGENT, "ofm-eod");
  curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
  if (const char* ca = std::getenv("CURL_CA_BUNDLE")) curl_easy_setopt(c, CURLOPT_CAINFO, ca);
  const CURLcode rc = curl_easy_perform(c);
  if (rc != CURLE_OK) r.error = curl_easy_strerror(rc);
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
  curl_easy_cleanup(c);
  return r;
}

// EODData's API: route plus parameters and the key (from EODDATA_API_KEY only; the URL holds it: never printed).
// Retries 429 and 5xx answers with a growing wait. Throws with the route and the status only.
inline json api(const std::string& route, const std::map<std::string, std::string>& params = {}, int tries = 5, int waitMs = 2000) {
  const char* key = std::getenv("EODDATA_API_KEY");
  const char* base = std::getenv("EODDATA_BASE");
  std::string url = std::string(base ? base : "https://api.eoddata.com") + route + "?";
  for (const auto& [k, v] : params) url += k + "=" + escape(v) + "&";
  url += "ApiKey=" + escape(key ? key : "");
  for (int attempt = 0;; ++attempt) {
    const Response r = get(url);
    if (r.status >= 200 && r.status < 300) return json::parse(r.body.empty() ? "[]" : r.body);
    if ((r.status == 429 || r.status >= 500 || r.status == 0) && attempt + 1 < tries) {
      std::this_thread::sleep_for(std::chrono::milliseconds(waitMs << attempt));
      continue;
    }
    throw std::runtime_error(route + ": HTTP " + std::to_string(r.status) + (r.error.empty() ? "" : " (" + r.error + ")"));
  }
}

struct Mail {
  std::string server, user, password, from, subject, text, html;
  std::vector<std::string> to;
  long port = 465;
};

inline std::size_t curlRead(char* p, std::size_t size, std::size_t n, void* user) {
  auto* s = static_cast<std::pair<std::string, std::size_t>*>(user);
  const std::size_t left = s->first.size() - s->second, k = std::min(left, size * n);
  std::memcpy(p, s->first.data() + s->second, k);
  s->second += k;
  return k;
}

// Sends a text-and-HTML message by SMTP: port 465 SSL, else STARTTLS. Returns "" or the error.
inline std::string sendMail(const Mail& m) {
  curlInit();
  const std::string boundary = "ofm-report-boundary";
  std::string body = "From: " + m.from + "\r\nTo: ";
  for (std::size_t k = 0; k < m.to.size(); ++k) body += (k ? ", " : "") + m.to[k];
  body += "\r\nSubject: " + m.subject + "\r\nMIME-Version: 1.0\r\nContent-Type: multipart/alternative; boundary=\"" + boundary + "\"\r\n\r\n";
  body += "--" + boundary + "\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n" + m.text + "\r\n";
  body += "--" + boundary + "\r\nContent-Type: text/html; charset=utf-8\r\n\r\n" + m.html + "\r\n--" + boundary + "--\r\n";
  std::pair<std::string, std::size_t> src{body, 0};
  CURL* c = curl_easy_init();
  const std::string url = (m.port == 465 ? "smtps://" : "smtp://") + m.server + ":" + std::to_string(m.port);
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  if (m.port != 465) curl_easy_setopt(c, CURLOPT_USE_SSL, long(CURLUSESSL_ALL));
  curl_easy_setopt(c, CURLOPT_USERNAME, m.user.c_str());
  curl_easy_setopt(c, CURLOPT_PASSWORD, m.password.c_str());
  curl_easy_setopt(c, CURLOPT_MAIL_FROM, ("<" + m.from + ">").c_str());
  curl_slist* rcpt = nullptr;
  for (const auto& t : m.to) rcpt = curl_slist_append(rcpt, ("<" + t + ">").c_str());
  curl_easy_setopt(c, CURLOPT_MAIL_RCPT, rcpt);
  curl_easy_setopt(c, CURLOPT_READFUNCTION, curlRead);
  curl_easy_setopt(c, CURLOPT_READDATA, &src);
  curl_easy_setopt(c, CURLOPT_UPLOAD, 1L);
  const CURLcode rc = curl_easy_perform(c);
  curl_slist_free_all(rcpt);
  curl_easy_cleanup(c);
  return rc == CURLE_OK ? "" : curl_easy_strerror(rc);
}

}  // namespace eod
