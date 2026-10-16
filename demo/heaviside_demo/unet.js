// Plain-JS forward pass of the demo U-Net (archs/unet1d.py with channels
// 16/32/64/128, kernel 5) on one field of length N. Weights come from
// model_weights.js, written by train_demo.py as float16 in state_dict order.
(function (root) {
  'use strict';

  function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function loadWeights(model) {
    const bin = atobAny(model.weights);
    const n = bin.length / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
    const W = {};
    let off = 0;
    for (const t of model.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      W[t.name] = { data: out.subarray(off, off + size), shape: t.shape };
      off += size;
    }
    return W;
  }

  function atobAny(s) {
    if (typeof atob === 'function') return atob(s);
    return Buffer.from(s, 'base64').toString('binary');
  }

  // x: Float32Array (cin * L); w: (cout, cin, k); same padding.
  function conv1d(x, cin, L, w, b) {
    const [cout, , k] = w.shape, pad = k >> 1, wd = w.data, y = new Float32Array(cout * L);
    if (k === 5) {  // hot path: zero-padded input, taps unrolled
      const P = L + 4, xp = new Float32Array(cin * P);
      for (let i = 0; i < cin; i++) xp.set(x.subarray(i * L, (i + 1) * L), i * P + 2);
      for (let o = 0; o < cout; o++) {
        const yb = o * L, bo = b.data[o];
        for (let t = 0; t < L; t++) y[yb + t] = bo;
        for (let i = 0; i < cin; i++) {
          const q = (o * cin + i) * 5, w0 = wd[q], w1 = wd[q + 1], w2 = wd[q + 2], w3 = wd[q + 3], w4 = wd[q + 4];
          const xb = i * P;
          for (let t = 0; t < L; t++) {
            const s = xb + t;
            y[yb + t] += w0 * xp[s] + w1 * xp[s + 1] + w2 * xp[s + 2] + w3 * xp[s + 3] + w4 * xp[s + 4];
          }
        }
      }
      return y;
    }
    for (let o = 0; o < cout; o++) {
      const yo = y.subarray(o * L, (o + 1) * L);
      yo.fill(b.data[o]);
      for (let i = 0; i < cin; i++) {
        const xi = x.subarray(i * L, (i + 1) * L);
        for (let j = 0; j < k; j++) {
          const wv = wd[(o * cin + i) * k + j];
          if (wv === 0) continue;
          const s = j - pad, t0 = Math.max(0, -s), t1 = Math.min(L, L - s);
          for (let t = t0; t < t1; t++) yo[t] += wv * xi[t + s];
        }
      }
    }
    return y;
  }

  function erf(x) {  // Abramowitz and Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1 - p * Math.exp(-x * x));
  }

  // GroupNorm(groups, C) followed by exact GELU, in place.
  function groupNormGelu(x, C, L, g, b, groups) {
    const cpg = C / groups, n = cpg * L;
    for (let gi = 0; gi < groups; gi++) {
      const base = gi * n;
      let mean = 0;
      for (let t = 0; t < n; t++) mean += x[base + t];
      mean /= n;
      let v = 0;
      for (let t = 0; t < n; t++) { const d = x[base + t] - mean; v += d * d; }
      const inv = 1 / Math.sqrt(v / n + 1e-5);
      for (let c = 0; c < cpg; c++) {
        const ch = gi * cpg + c, gw = g.data[ch] * inv, bw = b.data[ch] - mean * gw, o = ch * L;
        for (let t = 0; t < L; t++) {
          const z = x[o + t] * gw + bw;
          x[o + t] = 0.5 * z * (1 + erf(z * Math.SQRT1_2));
        }
      }
    }
    return x;
  }

  function convBlock(W, prefix, x, cin, L) {
    const w0 = W[prefix + '.net.0.weight'], cout = w0.shape[0];
    let h = conv1d(x, cin, L, w0, W[prefix + '.net.0.bias']);
    groupNormGelu(h, cout, L, W[prefix + '.net.1.weight'], W[prefix + '.net.1.bias'], 8);
    h = conv1d(h, cout, L, W[prefix + '.net.3.weight'], W[prefix + '.net.3.bias']);
    groupNormGelu(h, cout, L, W[prefix + '.net.4.weight'], W[prefix + '.net.4.bias'], 8);
    return h;
  }

  function maxPool2(x, C, L) {
    const h = L >> 1, y = new Float32Array(C * h);
    for (let c = 0; c < C; c++)
      for (let t = 0; t < h; t++) y[c * h + t] = Math.max(x[c * L + 2 * t], x[c * L + 2 * t + 1]);
    return y;
  }

  // ConvTranspose1d(cin, cout, 2, stride 2); w: (cin, cout, 2).
  function upConv(x, cin, L, w, b) {
    const cout = w.shape[1], L2 = 2 * L, wd = w.data, y = new Float32Array(cout * L2);
    for (let o = 0; o < cout; o++) {
      const yo = y.subarray(o * L2, (o + 1) * L2);
      yo.fill(b.data[o]);
      for (let i = 0; i < cin; i++) {
        const w0 = wd[(i * cout + o) * 2], w1 = wd[(i * cout + o) * 2 + 1], xi = x.subarray(i * L, (i + 1) * L);
        for (let t = 0; t < L; t++) { yo[2 * t] += w0 * xi[t]; yo[2 * t + 1] += w1 * xi[t]; }
      }
    }
    return y;
  }

  function concat(a, b) {
    const y = new Float32Array(a.length + b.length);
    y.set(a); y.set(b, a.length);
    return y;
  }

  // x: Float32Array (3 * N) with channels [linear interp, mask, observed*mask].
  // Returns the reconstruction: linear interp + U-Net residual.
  function forward(W, x, N, levels) {
    let h = x, C = 3, L = N;
    const skips = [];
    for (let i = 0; i < levels; i++) {
      h = convBlock(W, 'down.' + i, h, C, L);
      C = W['down.' + i + '.net.0.weight'].shape[0];
      skips.push([h, C]);
      h = maxPool2(h, C, L); L >>= 1;
    }
    h = convBlock(W, 'bottleneck', h, C, L);
    for (let i = 0; i < levels; i++) {
      const w = W['up.' + i + '.weight'];
      h = upConv(h, C, L, w, W['up.' + i + '.bias']);
      C = w.shape[1]; L <<= 1;
      const [skip, cs] = skips[levels - 1 - i];
      h = convBlock(W, 'dec.' + i, concat(h, skip), C + cs, L);
    }
    const res = conv1d(h, C, L, W['out.weight'], W['out.bias']);
    const y = new Float32Array(N);
    for (let t = 0; t < N; t++) y[t] = x[t] + res[t];
    return y;
  }

  root.DemoUNet = { loadWeights, forward };
  if (typeof module !== 'undefined') module.exports = root.DemoUNet;
})(typeof window !== 'undefined' ? window : globalThis);
