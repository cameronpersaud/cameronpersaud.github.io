// Anatomical orientation (same model as the desktop app).
// Frame: X = animal's left -> right, Y = back -> front, Z = bottom -> top.

export const PLANES = {
  coronal: {
    label: "Coronal",
    desc: "Cut front-to-back, like slicing bread from the nose towards the cerebellum. Each slice shows both hemispheres side by side.",
    ends: ["front (rostral)", "back (caudal)"],
    orders: ["Front → back (rostral → caudal)", "Back → front (caudal → rostral)"],
  },
  sagittal: {
    label: "Sagittal",
    desc: "Cut side-to-side, parallel to the midline. Each slice shows the brain in profile: nose at one end, cerebellum at the other.",
    ends: ["left side", "right side"],
    orders: ["Left → right", "Right → left"],
  },
  transverse: {
    label: "Transverse / horizontal",
    desc: "Cut top-to-bottom, parallel to the table. Each slice looks down on the brain, with both hemispheres and nose-to-cerebellum.",
    ends: ["top (dorsal)", "bottom (ventral)"],
    orders: ["Top → bottom (dorsal → ventral)", "Bottom → top (ventral → dorsal)"],
  },
};

const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];
const neg = (v) => v.map((x) => -x);

/**
 * o = { plane, topIsDorsal, rightIs: 'right'|'left'|'unknown', nosePoints: 'left'|'right', frontPoints: 'up'|'down' }
 * Returns 3x3 matrix (rows = world axes) whose COLUMNS are the world directions of
 * view axes: image right, image up, increasing slice number.
 */
export function matrix(o) {
  const sr = o.rightIs === "left" ? -1 : 1;
  const up = o.topIsDorsal ? Z : neg(Z);
  let cols;
  if (o.plane === "coronal") cols = [X.map((v) => v * sr), up, neg(Y)];
  else if (o.plane === "sagittal") cols = [o.nosePoints === "left" ? neg(Y) : Y, up, X];
  else cols = [X.map((v) => v * sr), o.frontPoints === "up" ? Y : neg(Y), neg(Z)];
  return [0, 1, 2].map((r) => cols.map((c) => c[r]));
}

export const lrKnown = (o) => o.plane === "sagittal" || o.rightIs !== "unknown";

export function axisTitles(o) {
  return [lrKnown(o) ? "Left → Right" : "Side A → Side B", "Back → Front", "Bottom → Top"];
}

/** World = M·view + t, with the view box [0, ext] landing in the positive octant. */
export function worldTransform(o, ext) {
  const M = matrix(o);
  let mn = [Infinity, Infinity, Infinity];
  for (const x of [0, ext[0]]) for (const y of [0, ext[1]]) for (const z of [0, ext[2]]) {
    const p = [0, 1, 2].map((r) => M[r][0] * x + M[r][1] * y + M[r][2] * z);
    mn = mn.map((v, i) => Math.min(v, p[i]));
  }
  return { M, t: mn.map((v) => -v) };
}

export function toWorld(p, M, t) { return [0, 1, 2].map((r) => M[r][0] * p[0] + M[r][1] * p[1] + M[r][2] * p[2] + t[r]); }

/** Image-based report numbers -> anatomical words. */
export function anatomicalPosition(rep, o) {
  if (!rep.found) return null;
  const [xl, yd, zf] = rep.centreFromEdges;
  const [sx, sy, sz] = rep.brainSize;
  const d = [xl, sy - yd, zf], size = [sx, sy, sz];
  const M = matrix(o);
  const pos = [0, 0, 0], len = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    let k = 0; for (let j = 1; j < 3; j++) if (Math.abs(M[a][j]) > Math.abs(M[a][k])) k = j;
    pos[a] = M[a][k] > 0 ? d[k] : size[k] - d[k];
    len[a] = size[k];
  }
  const [ml, ap, dv] = pos, [mlL, apL, dvL] = len;
  const off = ml - mlL / 2;
  const known = lrKnown(o);
  return {
    fromFront: apL - ap, fromFrontPct: (100 * (apL - ap)) / apL,
    fromTop: dvL - dv, fromTopPct: (100 * (dvL - dv)) / dvL,
    fromLeft: ml, fromLeftPct: (100 * ml) / mlL,
    midline: Math.abs(off), side: known ? (off > 0 ? "right" : "left") : (off > 0 ? "side B" : "side A"),
    length: apL, height: dvL, width: mlL, lrKnown: known,
  };
}

export function describe(o) {
  const p = PLANES[o.plane];
  return `${p.label} slices, slice 1 = ${p.ends[0]}, last = ${p.ends[1]}`;
}
