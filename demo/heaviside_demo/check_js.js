// Check the JS forward pass against the PyTorch reference stored in model_weights.js.
//   node demo/heaviside_demo/check_js.js
globalThis.window = globalThis;
require('./model_weights.js');
const { loadWeights, forward } = require('./unet.js');

const m = window.DEMO_MODEL, N = m.N;
const W = loadWeights(m);
const x = new Float32Array(3 * N);
m.test.x.forEach((row, c) => x.set(row, c * N));
const t0 = Date.now();
const y = forward(W, x, N, m.channels.length);
const ms = Date.now() - t0;
let maxErr = 0, maxRef = 0;
for (let i = 0; i < N; i++) {
  maxErr = Math.max(maxErr, Math.abs(y[i] - m.test.y[i]));
  maxRef = Math.max(maxRef, Math.abs(m.test.y[i]));
}
console.log(`max |js - torch| = ${maxErr.toExponential(2)} (max |y| = ${maxRef.toFixed(3)}), ${ms} ms`);
process.exit(maxErr < 1e-3 ? 0 : 1);
