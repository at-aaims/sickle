// JS port of sickle's MaxEnt sampler (subsampling/maxent.py) for a 1D field:
// k-means on eta, a histogram per cluster over the shared value range, pairwise
// KL divergence between cluster histograms, in-strength (column sum) per
// cluster, and per-point probability = in-strength of its cluster, normalized.
// sklearn's KMeans (k-means++, random_state 0) is replaced by Lloyd's
// algorithm from quantile centers, which is deterministic and converges to
// the same kind of contiguous value ranges in 1D.
(function (root) {
  'use strict';

  function kmeans1d(v, k, iters) {
    const n = v.length, sorted = Float64Array.from(v).sort();
    let c = [];
    for (let j = 0; j < k; j++) c.push(sorted[Math.min(n - 1, Math.floor((j + 0.5) / k * n))]);
    c = Array.from(new Set(c)).sort((a, b) => a - b);
    const labels = new Int32Array(n);
    for (let it = 0; it < (iters || 100); it++) {
      const kk = c.length, sum = new Float64Array(kk), cnt = new Int32Array(kk);
      let changed = false;
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let j = 0; j < kk; j++) { const d = Math.abs(v[i] - c[j]); if (d < bd) { bd = d; best = j; } }
        if (labels[i] !== best) { labels[i] = best; changed = true; }
        sum[best] += v[i]; cnt[best]++;
      }
      const nc = [];
      for (let j = 0; j < kk; j++) nc.push(cnt[j] ? sum[j] / cnt[j] : c[j]);
      c = nc;
      if (!changed && it > 0) break;
    }
    // drop empty clusters and relabel 0..K-1
    const counts = new Int32Array(c.length);
    for (let i = 0; i < n; i++) counts[labels[i]]++;
    const remap = new Int32Array(c.length).fill(-1);
    let K = 0;
    for (let j = 0; j < c.length; j++) if (counts[j]) remap[j] = K++;
    for (let i = 0; i < n; i++) labels[i] = remap[labels[i]];
    return { labels, K };
  }

  // Returns per-point probabilities (sum 1), cluster labels and in-strengths.
  function maxentProbs(eta, numClusters, numBins) {
    numClusters = numClusters || 10; numBins = numBins || 50;
    const n = eta.length;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, eta[i]); hi = Math.max(hi, eta[i]); }
    if (!(hi - lo > 1e-12)) return { probs: new Float64Array(n).fill(1 / n), labels: new Int32Array(n), inStrengths: [1], K: 1 };
    const { labels, K } = kmeans1d(eta, numClusters);
    // histograms like np.histogram(range=(lo, hi)): equal bins, last bin closed
    const hist = Array.from({ length: K }, () => new Float64Array(numBins));
    const tot = new Float64Array(K);
    for (let i = 0; i < n; i++) {
      const b = Math.min(numBins - 1, Math.floor((eta[i] - lo) / (hi - lo) * numBins));
      hist[labels[i]][b]++; tot[labels[i]]++;
    }
    // scipy.stats.entropy(p + 1e-10, q + 1e-10) renormalizes both before KL
    const P = hist.map((h, j) => {
      const p = h.map(x => x / tot[j] + 1e-10), s = p.reduce((a, b) => a + b, 0);
      return p.map(x => x / s);
    });
    const inStrengths = new Float64Array(K);
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        let kl = 0;
        for (let b = 0; b < numBins; b++) kl += P[i][b] * Math.log(P[i][b] / P[j][b]);
        inStrengths[j] += kl;   // column sum, as np.sum(adj_matrix, axis=0)
      }
    const probs = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { probs[i] = inStrengths[labels[i]]; s += probs[i]; }
    if (!(s > 0)) probs.fill(1 / n); else for (let i = 0; i < n; i++) probs[i] /= s;
    return { probs, labels, inStrengths: Array.from(inStrengths), K };
  }

  // Weighted sampling of m indices without replacement (Efraimidis-Spirakis
  // keys, equivalent to np.random.choice(replace=False, p=probs)).
  function weightedSample(probs, m, rnd) {
    const n = probs.length, keys = new Float64Array(n), order = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      keys[i] = probs[i] > 0 ? Math.log(Math.max(rnd(), 1e-300)) / probs[i] : -Infinity;
      order[i] = i;
    }
    order.sort((a, b) => keys[b] - keys[a]);
    return Array.from(order.subarray(0, Math.min(m, n))).sort((a, b) => a - b);
  }

  root.DemoMaxent = { maxentProbs, weightedSample, kmeans1d };
  if (typeof module !== 'undefined') module.exports = root.DemoMaxent;
})(typeof window !== 'undefined' ? window : globalThis);
