// Smooth 3D surfaces from stacks of 2D masks.
// Shape-based interpolation (signed distance per slice, interpolated between slices)
// + surface nets for the iso-surface + light Taubin smoothing.
import { edt, gaussianBlur } from "./imgproc.js";

/**
 * mask: Uint8Array Z×H×W. Returns { positions: Float32Array, indices: Uint32Array }
 * in VIEW coordinates (µm): x = image right, y = image up, z = slice direction.
 */
export function maskSurface(mask, Z, H, W, xyUm, zUm, maxPx = 220) {
  const step = Math.max(1, Math.ceil(Math.max(H, W) / maxPx));
  const h = Math.ceil(H / step), w = Math.ceil(W / step), sx = xyUm * step;
  const hp = h + 2, wp = w + 2, Zp = Z + 2, lim = 20;
  let any = false;
  // Signed distance per (padded) slice.
  const sd = [];
  for (let k = 0; k < Zp; k++) {
    const z = k - 1;
    const m = new Uint8Array(hp * wp);
    let has = false;
    if (z >= 0 && z < Z) {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        // a sub-sampled cell is inside if any of its pixels is
        let v = 0;
        for (let yy = y * step; yy < Math.min(H, (y + 1) * step) && !v; yy++)
          for (let xx = x * step; xx < Math.min(W, (x + 1) * step); xx++) if (mask[(z * H + yy) * W + xx]) { v = 1; break; }
        if (v) { m[(y + 1) * wp + x + 1] = 1; has = true; }
      }
    }
    if (!has) { sd.push(new Float32Array(hp * wp).fill(lim)); continue; }
    any = true;
    const inv = m.map((v) => 1 - v);
    const dOut = edt(m, wp, hp), dIn = edt(inv, wp, hp);
    const s = new Float32Array(hp * wp);
    for (let i = 0; i < s.length; i++) s[i] = Math.max(-lim, Math.min(lim, dOut[i] - dIn[i] + (m[i] ? 0.5 : -0.5)));
    sd.push(gaussianBlur(s, wp, hp, 1));
  }
  if (!any) return null;
  const up = Math.max(1, Math.min(4, Math.round(zUm / sx)));
  const nz = (Zp - 1) * up + 1, plane = hp * wp;
  const F = new Float32Array(nz * plane);
  for (let kz = 0; kz < nz; kz++) {
    const f = kz / up, k0 = Math.floor(f), k1 = Math.min(Zp - 1, k0 + 1), a = f - k0;
    const A = sd[k0], B = sd[k1];
    for (let i = 0; i < plane; i++) F[kz * plane + i] = A[i] * (1 - a) + B[i] * a;
  }
  const dz = zUm / up;
  const toView = (gx, gy, gz) => [(gx - 1) * sx, (H - 1) * xyUm - (gy - 1) * sx, gz * dz - zUm];
  const mesh = surfaceNets(F, wp, hp, nz, toView);
  taubin(mesh, 6);
  return mesh;
}

function surfaceNets(F, nx, ny, nz, toView) {
  const idx = (x, y, z) => (z * ny + y) * nx + x;
  const cubeVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cidx = (x, y, z) => (z * (ny - 1) + y) * (nx - 1) + x;
  const pos = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const v = new Float32Array(8);
  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    let inside = 0;
    for (let c = 0; c < 8; c++) { v[c] = F[idx(x + corners[c][0], y + corners[c][1], z + corners[c][2])]; if (v[c] < 0) inside++; }
    if (inside === 0 || inside === 8) continue;
    let px = 0, py = 0, pz = 0, cnt = 0;
    for (const [a, b] of edges) {
      if ((v[a] < 0) === (v[b] < 0)) continue;
      const t = v[a] / (v[a] - v[b]);
      px += corners[a][0] + t * (corners[b][0] - corners[a][0]);
      py += corners[a][1] + t * (corners[b][1] - corners[a][1]);
      pz += corners[a][2] + t * (corners[b][2] - corners[a][2]);
      cnt++;
    }
    cubeVert[cidx(x, y, z)] = pos.length / 3;
    pos.push(...toView(x + px / cnt, y + py / cnt, z + pz / cnt));
  }
  const tri = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) tri.push(a, c, b, a, d, c); else tri.push(a, b, c, a, c, d);
  };
  for (let z = 1; z < nz - 1; z++) for (let y = 1; y < ny - 1; y++) for (let x = 1; x < nx - 1; x++) {
    const s0 = F[idx(x, y, z)] < 0;
    if (x < nx - 1 && s0 !== (F[idx(x + 1, y, z)] < 0))
      quad(cubeVert[cidx(x, y - 1, z - 1)], cubeVert[cidx(x, y, z - 1)], cubeVert[cidx(x, y, z)], cubeVert[cidx(x, y - 1, z)], s0);
    if (y < ny - 1 && s0 !== (F[idx(x, y + 1, z)] < 0))
      quad(cubeVert[cidx(x - 1, y, z - 1)], cubeVert[cidx(x - 1, y, z)], cubeVert[cidx(x, y, z)], cubeVert[cidx(x, y, z - 1)], s0);
    if (z < nz - 1 && s0 !== (F[idx(x, y, z + 1)] < 0))
      quad(cubeVert[cidx(x - 1, y - 1, z)], cubeVert[cidx(x, y - 1, z)], cubeVert[cidx(x, y, z)], cubeVert[cidx(x - 1, y, z)], s0);
  }
  return { positions: Float32Array.from(pos), indices: Uint32Array.from(tri) };
}

function taubin(mesh, iters) {
  const P = mesh.positions, T = mesh.indices, n = P.length / 3;
  const nb = Array.from({ length: n }, () => new Set());
  for (let i = 0; i < T.length; i += 3) {
    const [a, b, c] = [T[i], T[i + 1], T[i + 2]];
    nb[a].add(b); nb[a].add(c); nb[b].add(a); nb[b].add(c); nb[c].add(a); nb[c].add(b);
  }
  const lists = nb.map((s) => Int32Array.from(s));
  const tmp = new Float32Array(P.length);
  const pass = (lam) => {
    for (let i = 0; i < n; i++) {
      const L = lists[i];
      if (!L.length) { tmp[3 * i] = P[3 * i]; tmp[3 * i + 1] = P[3 * i + 1]; tmp[3 * i + 2] = P[3 * i + 2]; continue; }
      let ax = 0, ay = 0, az = 0;
      for (const j of L) { ax += P[3 * j]; ay += P[3 * j + 1]; az += P[3 * j + 2]; }
      ax /= L.length; ay /= L.length; az /= L.length;
      tmp[3 * i] = P[3 * i] + lam * (ax - P[3 * i]);
      tmp[3 * i + 1] = P[3 * i + 1] + lam * (ay - P[3 * i + 1]);
      tmp[3 * i + 2] = P[3 * i + 2] + lam * (az - P[3 * i + 2]);
    }
    P.set(tmp);
  };
  for (let k = 0; k < iters; k++) { pass(0.5); pass(-0.53); }
}

/** Binary STL from a mesh already in world coordinates. */
export function toSTL(positions, indices, name = "brainstack") {
  const nTri = indices.length / 3;
  const buf = new ArrayBuffer(84 + nTri * 50), dv = new DataView(buf);
  const hdr = `BrainStack 3D ${name} (units: micrometres)`;
  for (let i = 0; i < Math.min(80, hdr.length); i++) dv.setUint8(i, hdr.charCodeAt(i));
  dv.setUint32(80, nTri, true);
  let o = 84;
  for (let t = 0; t < nTri; t++) {
    const a = indices[3 * t] * 3, b = indices[3 * t + 1] * 3, c = indices[3 * t + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
    for (const v of [nx, ny, nz]) { dv.setFloat32(o, v, true); o += 4; }
    for (const p of [a, b, c]) for (let k = 0; k < 3; k++) { dv.setFloat32(o, positions[p + k], true); o += 4; }
    o += 2;
  }
  return new Blob([buf], { type: "model/stl" });
}
