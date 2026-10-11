// #include header.wgsl
// The self-adaptive selector (one invocation; after book). At each close t it scores three
// candidates on their own net records so far: 0 the learnt policy (the booked top 10), 1 the top 10
// without stops, 2 the market bought and held; cash scores 0. Each record is an ADWIN window
// (Bifet and Gavalda 2007; scale-free bound, confidence 1/T: one false cut expected over the whole
// history; each side at least the square root of the window) over the candidate's daily net
// returns: it grows while the record is stable and drops its older part when it changes. The score
// is the window's mean return shrunk by its own t-statistic, m * max(0, 1 - 1/t^2), as the factor
// premia are, so a record no better than noise scores 0 and cash stays. The highest score is
// followed on day t + 1; moving into or out of the market pays the dealing costs.
// days[t][21]: the choice (0 cash, 1 learnt policy, 2 no stops, 3 market); [22]: the selector's net
// return on day t + 1; [23 + c]: candidate c's score.
@group(0) @binding(10) var<storage, read_write> days : array<f32>;
@group(0) @binding(12) var<storage, read> mkt : array<f32>;
@group(0) @binding(13) var<storage, read_write> sel : array<f32>;

const NC : u32 = 3u;
var<private> W0 : array<u32, 3>;
var<private> CNT : u32;

fn rec(c : u32, k : u32, j : u32) -> f32 { return sel[(c * (P.T + 1u) + k) * 2u + j]; }
fn put(c : u32, k : u32, j : u32, v : f32) { sel[(c * (P.T + 1u) + k) * 2u + j] = v; }

// Candidate c's net return on day t + 1 (row t of the book).
fn candidate(c : u32, t : u32) -> f32 {
  if (c == 0u) { return days[t * KD]; }
  if (c == 1u) { return days[t * KD + 2u]; }
  return mkt[t + 1u];
}

fn cut(c : u32) {
  loop {
    let a = W0[c];
    let n = CNT - a;
    let side = max(2u, u32(ceil(sqrt(f32(n)))));
    if (n < 2u * side) { return; }
    let lt = log(2.0 * log(f32(n)) * f32(P.T));
    var moved = false;
    for (var k = a + side; k + side <= CNT; k = k + 1u) {
      let n0 = f32(k - a);
      let n1 = f32(CNT - k);
      let m0 = (rec(c, k, 0u) - rec(c, a, 0u)) / n0;
      let m1 = (rec(c, CNT, 0u) - rec(c, k, 0u)) / n1;
      let v0 = max(0.0, (rec(c, k, 1u) - rec(c, a, 1u)) / n0 - m0 * m0);
      let v1 = max(0.0, (rec(c, CNT, 1u) - rec(c, k, 1u)) / n1 - m1 * m1);
      if (abs(m0 - m1) > sqrt(2.0 * lt * (v0 / n0 + v1 / n1))) {
        W0[c] = k;
        moved = true;
        break;
      }
    }
    if (!moved) { return; }
  }
}

fn score(c : u32) -> f32 {
  let a = W0[c];
  let n = CNT - a;
  if (n < 2u) { return 0.0; }
  let fn1 = f32(n);
  let m = (rec(c, CNT, 0u) - rec(c, a, 0u)) / fn1;
  let v = max(0.0, ((rec(c, CNT, 1u) - rec(c, a, 1u)) - fn1 * m * m) / (fn1 - 1.0));
  if (!(v > 0.0)) { return 0.0; }
  let t2 = m * m / (v / fn1);
  return m * max(0.0, 1.0 - 1.0 / t2);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u) { return; }
  CNT = 0u;
  for (var c = 0u; c < NC; c = c + 1u) { W0[c] = 0u; put(c, 0u, 0u, 0.0); put(c, 0u, 1u, 0.0); }
  var prev = 0u;
  for (var t = P.s; t < P.T; t = t + 1u) {
    if (t > P.s) {
      for (var c = 0u; c < NC; c = c + 1u) {
        let v = candidate(c, t - 1u);
        put(c, CNT + 1u, 0u, rec(c, CNT, 0u) + v);
        put(c, CNT + 1u, 1u, rec(c, CNT, 1u) + v * v);
      }
      CNT = CNT + 1u;
      for (var c = 0u; c < NC; c = c + 1u) { cut(c); }
    }
    var best = 0.0;
    var choice = 0u;
    for (var c = 0u; c < NC; c = c + 1u) {
      let s = score(c);
      days[t * KD + 23u + c] = s;
      if (s > best) { best = s; choice = c + 1u; }
    }
    days[t * KD + 21u] = f32(choice);
    var r = 0.0;
    if (t + 1u < P.T) {
      if (choice > 0u) { r = candidate(choice - 1u, t); }
      if (choice == 3u && prev != 3u) { r = r - P.buy; }
      if (prev == 3u && choice != 3u) { r = r - P.sell; }
    }
    days[t * KD + 22u] = r;
    prev = choice;
  }
}
