// The private store (svr-net/eod) through libgit2: the clone at data/eoddata brought up to date (sync) and what a
// run wrote committed and pushed (push), never one file at a time through GitHub's API and never with the git
// command line. Authentication: EOD_TOKEN as the HTTPS password (x-access-token).
//
//   sync  open the clone (or make it: depth 1), fetch main's newest commit, the working copy reset to it (the files
//         the run will read and add to)
//   push  every change in the working copy (new, changed and deleted files) in one commit on main, pushed (never
//         forced); when another market's job pushed first, the run's changes are made again on top of the newer
//         commit (a file both changed keeps this run's version, as `git pull --rebase -X ours` did), at most 5 tries
#pragma once

#include <git2.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "eod/common.hpp"

namespace eod::git {

struct Lib {
  Lib() { git_libgit2_init(); }
  ~Lib() { git_libgit2_shutdown(); }
};
inline void init() { static Lib lib; }

inline void check(int rc, const char* what) {
  if (rc < 0) {
    const git_error* e = git_error_last();
    throw std::runtime_error(std::string(what) + ": " + (e && e->message ? e->message : "libgit2 error " + std::to_string(rc)));
  }
}

struct Push {
  std::string refused;
};

inline int credentials(git_credential** out, const char*, const char*, unsigned int, void*) {
  const char* t = std::getenv("EOD_TOKEN");
  if (!t || !*t) return GIT_PASSTHROUGH;
  return git_credential_userpass_plaintext_new(out, "x-access-token", t);
}

inline int pushUpdate(const char* ref, const char* status, void* payload) {
  if (status) static_cast<Push*>(payload)->refused = std::string(ref) + ": " + status;
  return 0;
}

inline git_fetch_options fetchOptions(int depth) {
  git_fetch_options o;
  git_fetch_options_init(&o, GIT_FETCH_OPTIONS_VERSION);
  o.callbacks.credentials = credentials;
  o.depth = depth;
  return o;
}

// The newest commit of origin's main (after a fetch).
inline git_oid originMain(git_repository* r) {
  git_oid id;
  check(git_reference_name_to_id(&id, r, "refs/remotes/origin/main"), "origin/main");
  return id;
}

// The working copy, index and main reset to a commit (a hard reset: the files the run reads).
inline void resetTo(git_repository* r, const git_oid& id) {
  git_object* obj = nullptr;
  check(git_object_lookup(&obj, r, &id, GIT_OBJECT_COMMIT), "lookup");
  git_checkout_options co;
  git_checkout_options_init(&co, GIT_CHECKOUT_OPTIONS_VERSION);
  co.checkout_strategy = GIT_CHECKOUT_FORCE | GIT_CHECKOUT_REMOVE_UNTRACKED;
  git_reference* ref = nullptr;
  check(git_reference_create(&ref, r, "refs/heads/main", &id, 1, "ofm-eod: main"), "main");
  git_reference_free(ref);
  check(git_repository_set_head(r, "refs/heads/main"), "head");
  check(git_reset(r, obj, GIT_RESET_HARD, &co), "reset");
  git_object_free(obj);
}

inline void fetchMain(git_repository* r, int depth) {
  git_remote* remote = nullptr;
  check(git_remote_lookup(&remote, r, "origin"), "origin");
  git_fetch_options fo = fetchOptions(depth);
  char spec[] = "+refs/heads/main:refs/remotes/origin/main";
  char* specs[] = {spec};
  git_strarray refspecs{specs, 1};
  const int rc = git_remote_fetch(remote, &refspecs, &fo, "ofm-eod fetch");
  git_remote_free(remote);
  check(rc, "fetch svr-net/eod");
}

// The clone at dir up to date with main's newest commit (made when missing). Returns the commit's short id and date.
inline std::string sync(const std::string& dir, const std::string& url, int depth = 1) {
  init();
  git_repository* r = nullptr;
  if (git_repository_open(&r, dir.c_str()) < 0) {
    git_clone_options co;
    git_clone_options_init(&co, GIT_CLONE_OPTIONS_VERSION);
    co.fetch_opts = fetchOptions(depth);
    co.checkout_branch = "main";
    check(git_clone(&r, url.c_str(), dir.c_str(), &co), "clone svr-net/eod");
  } else {
    fetchMain(r, depth);
    resetTo(r, originMain(r));
  }
  git_oid head;
  check(git_reference_name_to_id(&head, r, "HEAD"), "HEAD");
  git_commit* c = nullptr;
  check(git_commit_lookup(&c, r, &head), "commit");
  char sha[8];
  git_oid_tostr(sha, sizeof sha, &head);
  const std::string when = ymd(std::time_t(git_commit_time(c)));
  git_commit_free(c);
  git_repository_free(r);
  return std::string(sha) + " of " + when;
}

// Every change of the working copy in one commit on top of `parent` (or the index's tree when none changed).
inline bool commitWorkingCopy(git_repository* r, const std::string& message, git_oid& out, git_oid& parentId) {
  git_index* idx = nullptr;
  check(git_repository_index(&idx, r), "index");
  char star[] = "*";
  char* pats[] = {star};
  git_strarray all{pats, 1};
  check(git_index_add_all(idx, &all, GIT_INDEX_ADD_DEFAULT, nullptr, nullptr), "add");
  check(git_index_update_all(idx, &all, nullptr, nullptr), "update");
  check(git_index_write(idx), "index write");
  git_oid tree;
  check(git_index_write_tree(&tree, idx), "tree");
  git_index_free(idx);
  check(git_reference_name_to_id(&parentId, r, "HEAD"), "HEAD");
  git_commit* parent = nullptr;
  check(git_commit_lookup(&parent, r, &parentId), "parent");
  if (git_oid_equal(git_commit_tree_id(parent), &tree)) {
    git_commit_free(parent);
    return false;  // nothing new
  }
  git_tree* t = nullptr;
  check(git_tree_lookup(&t, r, &tree), "tree");
  git_signature* sig = nullptr;
  check(git_signature_now(&sig, "github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com"), "signature");
  const git_commit* parents[] = {parent};
  check(git_commit_create(&out, r, "HEAD", sig, sig, nullptr, message.c_str(), t, 1, parents), "commit");
  git_signature_free(sig);
  git_tree_free(t);
  git_commit_free(parent);
  return true;
}

// The run's changes (ours: its commit against its parent) made again on top of `base`: a new commit.
inline git_oid replay(git_repository* r, const git_oid& ours, const git_oid& base, const std::string& message) {
  git_commit *mine = nullptr, *mineParent = nullptr, *baseCommit = nullptr;
  check(git_commit_lookup(&mine, r, &ours), "ours");
  check(git_commit_parent(&mineParent, mine, 0), "parent");
  check(git_commit_lookup(&baseCommit, r, &base), "base");
  git_tree *a = nullptr, *b = nullptr, *baseTree = nullptr;
  check(git_commit_tree(&a, mineParent), "tree");
  check(git_commit_tree(&b, mine), "tree");
  check(git_commit_tree(&baseTree, baseCommit), "tree");
  git_diff* diff = nullptr;
  check(git_diff_tree_to_tree(&diff, r, a, b, nullptr), "diff");
  std::vector<git_tree_update> ups;
  std::vector<std::string> paths;
  const std::size_t n = git_diff_num_deltas(diff);
  paths.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    const git_diff_delta* dd = git_diff_get_delta(diff, k);
    const bool gone = dd->status == GIT_DELTA_DELETED;
    paths.push_back(gone ? dd->old_file.path : dd->new_file.path);
  }
  for (std::size_t k = 0; k < n; ++k) {
    const git_diff_delta* dd = git_diff_get_delta(diff, k);
    git_tree_update u{};
    if (dd->status == GIT_DELTA_DELETED) {
      u.action = GIT_TREE_UPDATE_REMOVE;
    } else {
      u.action = GIT_TREE_UPDATE_UPSERT;
      u.id = dd->new_file.id;
      u.filemode = git_filemode_t(dd->new_file.mode);
    }
    u.path = paths[k].c_str();
    ups.push_back(u);
  }
  git_oid treeId;
  check(git_tree_create_updated(&treeId, r, baseTree, ups.size(), ups.data()), "tree on top");
  git_tree* t = nullptr;
  check(git_tree_lookup(&t, r, &treeId), "tree");
  git_signature* sig = nullptr;
  check(git_signature_now(&sig, "github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com"), "signature");
  const git_commit* parents[] = {baseCommit};
  git_oid out;
  check(git_commit_create(&out, r, nullptr, sig, sig, nullptr, message.c_str(), t, 1, parents), "commit");
  git_signature_free(sig);
  git_tree_free(t), git_tree_free(a), git_tree_free(b), git_tree_free(baseTree);
  git_diff_free(diff);
  git_commit_free(mine), git_commit_free(mineParent), git_commit_free(baseCommit);
  return out;
}

inline std::string pushMain(git_repository* r) {
  git_remote* remote = nullptr;
  check(git_remote_lookup(&remote, r, "origin"), "origin");
  git_push_options po;
  git_push_options_init(&po, GIT_PUSH_OPTIONS_VERSION);
  po.callbacks.credentials = credentials;
  Push result;
  po.callbacks.push_update_reference = pushUpdate;
  po.callbacks.payload = &result;
  char spec[] = "refs/heads/main:refs/heads/main";
  char* specs[] = {spec};
  git_strarray refspecs{specs, 1};
  const int rc = git_remote_push(remote, &refspecs, &po);
  git_remote_free(remote);
  if (rc < 0) {
    const git_error* e = git_error_last();
    return e && e->message ? e->message : "push failed";
  }
  return result.refused;
}

// Commits and pushes the working copy's changes. Returns what happened, in words.
inline std::string push(const std::string& dir, const std::string& message, int depth = 1) {
  init();
  git_repository* r = nullptr;
  check(git_repository_open(&r, dir.c_str()), "open the store");
  git_oid ours, parent;
  if (!commitWorkingCopy(r, message, ours, parent)) {
    git_repository_free(r);
    return "nothing new";
  }
  std::string last;
  for (int k = 0; k < 5; ++k) {
    last = pushMain(r);
    if (last.empty()) {
      git_repository_free(r);
      return k ? "pushed (made again on top of " + std::to_string(k) + " newer commit" + (k > 1 ? "s" : "") + ")" : "pushed";
    }
    std::this_thread::sleep_for(std::chrono::seconds(2 * (k + 1)));
    fetchMain(r, depth);  // someone pushed first: this run's changes again on top of theirs
    const git_oid next = replay(r, ours, originMain(r), message);
    resetTo(r, next);
    ours = next;
  }
  git_repository_free(r);
  throw std::runtime_error("push to svr-net/eod refused 5 times (" + last + ")");
}

}  // namespace eod::git
