// #include header.wgsl
// The self-adaptive selector (one invocation; after book). At each close t it scores four
// candidates on their own net records so far: 0 the learnt policy (the booked top 10), 1 the top 10
// without stops, 2 the market bought and held, 3 the model's own portfolio (held over several days,
// traded only when a forecast beats the costs); cash scores 0. Each record is an ADWIN window
// (Bifet and Gavalda 2007; scale-free bound, confidence 1/T: one false cut expected over the whole
// history; each side at least the square root of the window) over the candidate's daily net
// returns: it grows while the record is stable and drops its older part when it changes. The score
// is the window's mean return shrunk by its own t-statistic, m * max(0, 1 - 1/t^2), as the factor
// premia are, so a record no better than noise scores 0 and cash stays. The highest score is
// followed on day t + 1; moving into a held portfolio (market, model) pays the purchase cost on the
// whole book, moving out of one the sale cost (the day trades pay their own round trips).
// days[t][21]: the choice (0 cash, 1 learnt policy, 2 no stops, 3 market, 4 model portfolio); [22]:
// the selector's net return on day t + 1; [23 + c]: candidate c's score.
@group(0) @binding(10) var<storage, read_write> days : array<f32>;
@group(0) @binding(12) var<storage, read> ext : array<f32>;  // [2][T]: the market's and the model portfolio's net return on day d
@group(0) @binding(13) var<storage, read_write> sel : array<f32>;

const NC : u32 = 4u;
var<private> W0 : array<u32, 4>;
var<private> CNT : u32;

fn rec(c : u32, k : u32, j : u32) -> f32 { return sel[(c * (P.T + 1u) + k) * 2u + j]; }
fn put(c : u32, k : u32, j : u32, v : f32) { sel[(c * (P.T + 1u) + k) * 2u + j] = v; }

// Candidate c's net return on day t + 1 (row t of the book).
fn candidate(c : u32, t : u32) -> f32 {
  if (c == 0u) { return days[t * KD]; }
  if (c == 1u) { return days[t * KD + 2u]; }
  return ext[(c - 2u) * P.T + t + 1u];
}

fn held(choice : u32) -> bool { return choice == 3u || choice == 4u; }

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
      if (c == 3u && s > best) { best = s; choice = c + 1u; }  // experiment: the model portfolio or cash only
    }
    days[t * KD + 21u] = f32(choice);
    var r = 0.0;
    if (t + 1u < P.T) {
      if (choice > 0u) { r = candidate(choice - 1u, t); }
      if (held(choice) && choice != prev) { r = r - P.buy; }
      if (held(prev) && choice != prev) { r = r - P.sell; }
    }
    days[t * KD + 22u] = r;
    prev = choice;
  }
}
