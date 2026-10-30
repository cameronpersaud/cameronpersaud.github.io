// three.js scene: see-through brain + dye in the anatomical frame.
import { toWorld, worldTransform } from "./orientation.js";

const VIEWS = { "3d": [-1.0, 1.3, 0.9], side: [-1, 0, 0], top: [0, 0, 1], front: [0, 1, 0] };

export class Viewer {
  constructor(canvas, compassCanvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 10, 500000);
    this.camera.up.set(0, 0, 1);
    this.controls = new THREE.OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(-1, 1.3, 1.6); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(1, -1, -0.5); this.scene.add(fill);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.compass = compassCanvas ? compassCanvas.getContext("2d") : null;
    this.spin = false;
    this.center = new THREE.Vector3(); this.size = new THREE.Vector3(1, 1, 1);
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    if (!r.width || !r.height) return;
    this.renderer.setSize(r.width, r.height, false);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    requestAnimationFrame(this._loop);
    if (this.spin) {
      const off = this.camera.position.clone().sub(this.controls.target);
      off.applyAxisAngle(new THREE.Vector3(0, 0, 1), 0.004);
      this.camera.position.copy(this.controls.target).add(off);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._drawCompass();
  }

  _drawCompass() {
    const cc = this.compass; if (!cc) return;
    const w = cc.canvas.width, c = w / 2;
    cc.clearRect(0, 0, w, w);
    const m = new THREE.Matrix4().extractRotation(this.camera.matrixWorldInverse);
    cc.font = `500 ${Math.round(w / 9)}px "IBM Plex Mono", monospace`;
    cc.textAlign = "center"; cc.textBaseline = "middle";
    for (const [v, label, col] of [[[0, 1, 0], "Front", "#ff5a5a"], [[0, 0, 1], "Top", "#b9c7d6"]]) {
      const p = new THREE.Vector3(...v).applyMatrix4(m);
      cc.strokeStyle = col; cc.fillStyle = col; cc.lineWidth = 3;
      cc.beginPath(); cc.moveTo(c, c); cc.lineTo(c + p.x * w * 0.24, c - p.y * w * 0.24); cc.stroke();
      cc.fillText(label, c + p.x * w * 0.36, c - p.y * w * 0.36);
    }
  }

  /** brain/dye meshes in VIEW coords; extent = view box size. */
  setScene({ brain, dye, centre, extent, orientation }) {
    this.root.clear();
    const { M, t } = worldTransform(orientation, extent);
    this.M = M; this.t = t;
    const m4 = new THREE.Matrix4().set(
      M[0][0], M[0][1], M[0][2], t[0], M[1][0], M[1][1], M[1][2], t[1], M[2][0], M[2][1], M[2][2], t[2], 0, 0, 0, 1);
    const holder = new THREE.Group(); holder.matrixAutoUpdate = false; holder.matrix.copy(m4);
    this.root.add(holder);
    const geom = (m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
      g.setIndex(new THREE.BufferAttribute(m.indices, 1));
      g.computeVertexNormals();
      return g;
    };
    this.brainMesh = this.dyeMesh = null;
    if (dye) {
      this.dyeMesh = new THREE.Mesh(geom(dye), new THREE.MeshPhongMaterial({
        color: 0xff4040, emissive: 0x3a0505, transparent: true, opacity: 0.95, side: THREE.DoubleSide, shininess: 60 }));
      this.dyeMesh.renderOrder = 1; holder.add(this.dyeMesh);
    }
    if (brain) {
      this.brainMesh = new THREE.Mesh(geom(brain), new THREE.MeshPhongMaterial({
        color: 0xb9c7d6, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide, shininess: 20 }));
      this.brainMesh.renderOrder = 2; holder.add(this.brainMesh);
    }
    // Bounds in world coordinates
    const ex = extent, corners = [];
    for (const x of [0, ex[0]]) for (const y of [0, ex[1]]) for (const z of [0, ex[2]]) corners.push(toWorld([x, y, z], M, t));
    const box = new THREE.Box3(); corners.forEach((p) => box.expandByPoint(new THREE.Vector3(...p)));
    if (brain) { holder.updateMatrixWorld(true); box.setFromObject(this.brainMesh); }
    this.size = box.getSize(new THREE.Vector3()); this.center = box.getCenter(new THREE.Vector3());
    const r = Math.max(this.size.x, this.size.y, this.size.z);
    this.centreDot = null;
    if (centre) {
      this.centreDot = new THREE.Mesh(new THREE.SphereGeometry(r * 0.012, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffd23f }));
      this.centreDot.position.set(...toWorld(centre, M, t));
      this.root.add(this.centreDot);
    }
    this.grid = this._grid(); this.root.add(this.grid);
    this.resize();
  }

  _grid() {
    // Floor grid in 1 mm squares: a real scale reference.
    const c = this.center, s = this.size, pts = [];
    const x0 = Math.floor((c.x - s.x * 0.7) / 1000) * 1000, x1 = Math.ceil((c.x + s.x * 0.7) / 1000) * 1000;
    const y0 = Math.floor((c.y - s.y * 0.7) / 1000) * 1000, y1 = Math.ceil((c.y + s.y * 0.7) / 1000) * 1000;
    const z = c.z - s.z / 2 - 300;
    for (let x = x0; x <= x1; x += 1000) pts.push(x, y0, z, x, y1, z);
    for (let y = y0; y <= y1; y += 1000) pts.push(x0, y, z, x1, y, z);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x2a323c }));
  }

  setView(name) {
    const d = new THREE.Vector3(...VIEWS[name]).normalize();
    const dist = Math.max(this.size.x, this.size.y, this.size.z) * 2.1;
    this.camera.up.set(0, 0, 1);
    if (name === "top") this.camera.up.set(0, 1, 0);
    this.camera.position.copy(this.center).addScaledVector(d, dist);
    this.controls.target.copy(this.center);
    this.controls.update();
  }

  setOpacity(which, v) {
    const m = which === "brain" ? this.brainMesh : this.dyeMesh;
    if (!m) return;
    m.material.opacity = v; m.visible = v > 0.001;
  }

  worldMesh(which) {
    const m = which === "brain" ? this.brainMesh : this.dyeMesh;
    if (!m) return null;
    const P = m.geometry.attributes.position.array, out = new Float32Array(P.length);
    for (let i = 0; i < P.length; i += 3) out.set(toWorld([P[i], P[i + 1], P[i + 2]], this.M, this.t), i);
    return { positions: out, indices: m.geometry.index.array };
  }

  snapshot() { this.renderer.render(this.scene, this.camera); return this.canvas.toDataURL("image/png"); }
}
