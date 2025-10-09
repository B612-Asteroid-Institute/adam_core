import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(() => {
  const container = document.getElementById('container');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);

  const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 1e6);
  camera.position.set(5, 5, 5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  const axes = new THREE.AxesHelper(2);
  scene.add(axes);

  const grid = new THREE.GridHelper(10, 10, 0x1e90ff, 0x16202b);
  grid.rotation.x = Math.PI / 2; // x-y plane for AU context
  scene.add(grid);

  // Layer groups
  const groupSegments = new THREE.Group();
  const groupBoxes = new THREE.Group();
  const groupRays = new THREE.Group();
  scene.add(groupSegments, groupBoxes, groupRays);

  // UI elements
  const fileInput = document.getElementById('fileInput');
  const cbSegments = document.getElementById('cbSegments');
  const cbBoxes = document.getElementById('cbBoxes');
  const cbRays = document.getElementById('cbRays');
  const cbHits = document.getElementById('cbHits');
  const cbMisses = document.getElementById('cbMisses');
  const depthSlider = document.getElementById('depthSlider');
  const depthValue = document.getElementById('depthValue');
  const rayLen = document.getElementById('rayLen');
  const rayLenValue = document.getElementById('rayLenValue');
  const resetViewBtn = document.getElementById('resetViewBtn');
  const colorByStation = document.getElementById('colorByStation');

  let data = null;
  let maxDepth = 0;

  function getRayLengthAU() {l
    let v = parseFloat(rayLen.value);
    if (!Number.isFinite(v)) v = 3;
    if (v < 0.1) v = 0.1;
    if (v > 40) v = 40;
    rayLen.value = String(v);
    rayLenValue.textContent = v.toFixed(1);
    return v;
  }

  function updateRaysOnly() {
    if (!data) return;
    const hitMask = Array.isArray(data.rays_hit_mask) ? data.rays_hit_mask : (data.rays_hit_mask || null);
    buildRays(
      data.rays_origins_f32,
      data.rays_dirs_f32,
      getRayLengthAU(),
      hitMask,
      data.rays_station_codes || null,
      colorByStation.checked,
    );
  }

  function fitToBoundsSphere(bounds) {
    if (!bounds) return;
    const [cx, cy, cz, r] = bounds;
    const center = new THREE.Vector3(cx, cy, cz);
    const dist = r * 2.5;
    camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist, dist)));
    camera.near = Math.max(0.001, r * 0.01);
    camera.far = Math.max(1000, r * 20);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }

  function clearGroup(group) {
    while (group.children.length) {
      const obj = group.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }
  }

  function buildSegments(segments) {
    clearGroup(groupSegments);
    if (!segments || segments.length === 0) return;
    const positions = new Float32Array(segments.flat());
    const geom = new THREE.BufferGeometry();
    // positions as pairs of points: 6 floats per segment
    const pos = new Float32BufferAttributeSafe(positions, 3);
    geom.setAttribute('position', pos);
    const mat = new THREE.LineBasicMaterial({ color: 0x7cc7ff, linewidth: 1 });
    const lines = new THREE.LineSegments(geom, mat);
    groupSegments.add(lines);
  }

  function buildBoxes(nodesMin, nodesMax) {
    clearGroup(groupBoxes);
    if (!nodesMin || !nodesMax || nodesMin.length === 0) return;
    const color = 0x6ee7b7;
    const material = new THREE.LineBasicMaterial({ color, linewidth: 1 });
    for (let i = 0; i < nodesMin.length; i++) {
      const mn = nodesMin[i];
      const mx = nodesMax[i];
      const geom = new THREE.BufferGeometry();
      const verts = boxWireframeVertices(mn, mx);
      geom.setAttribute('position', new Float32BufferAttributeSafe(new Float32Array(verts), 3));
      const lines = new THREE.LineSegments(geom, material);
      lines.userData.depth = 0; // filled by filter
      groupBoxes.add(lines);
    }
  }

  function filterBoxesByDepth(depths, maxAllowedDepth) {
    if (!depths) return;
    for (let i = 0; i < groupBoxes.children.length; i++) {
      const obj = groupBoxes.children[i];
      const visible = depths[i] <= maxAllowedDepth;
      obj.visible = visible;
    }
  }

  function buildRays(origins, dirs, lengthAU, hitMask = null, stationCodes = null, colorByStationFlag = false) {
    clearGroup(groupRays);
    if (!origins || origins.length === 0) return;
    const positions = new Float32Array(origins.length * 2 * 3);
    for (let i = 0; i < origins.length; i++) {
      const o = origins[i];
      const d = dirs && dirs[i] ? dirs[i] : [0, 0, 1];
      const x0 = o[0], y0 = o[1], z0 = o[2];
      const x1 = x0 + d[0] * lengthAU;
      const y1 = y0 + d[1] * lengthAU;
      const z1 = z0 + d[2] * lengthAU;
      const base = i * 6;
      positions[base + 0] = x0; positions[base + 1] = y0; positions[base + 2] = z0;
      positions[base + 3] = x1; positions[base + 4] = y1; positions[base + 5] = z1;
    }
    // Render as standard 1px GL line segments (no apparent width in world space)
    // If hitMask filtering is requested
    let maskHits = null, maskMisses = null;
    if (hitMask && hitMask.length === (origins.length)) {
      maskHits = new Array(origins.length).fill(false);
      maskMisses = new Array(origins.length).fill(false);
      for (let i = 0; i < origins.length; i++) maskHits[i] = !!hitMask[i];
      for (let i = 0; i < origins.length; i++) maskMisses[i] = !maskHits[i];
    }

    // Helper to build a subset by boolean selector with a color
    function addSubset(selector, color) {
      if (!selector) {
        // no selector: draw all in one material
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new Float32BufferAttributeSafe(positions, 3));
        const mat = new THREE.LineBasicMaterial({ color });
        groupRays.add(new THREE.LineSegments(geom, mat));
        return;
      }
      let count = 0;
      for (let i = 0; i < origins.length; i++) if (selector[i]) count++;
      if (count === 0) return;
      const sub = new Float32Array(count * 2 * 3);
      let w = 0;
      for (let i = 0; i < origins.length; i++) {
        if (!selector[i]) continue;
        const base = i * 6;
        sub[w + 0] = positions[base + 0];
        sub[w + 1] = positions[base + 1];
        sub[w + 2] = positions[base + 2];
        sub[w + 3] = positions[base + 3];
        sub[w + 4] = positions[base + 4];
        sub[w + 5] = positions[base + 5];
        w += 6;
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new Float32BufferAttributeSafe(sub, 3));
      const mat = new THREE.LineBasicMaterial({ color });
      groupRays.add(new THREE.LineSegments(geom, mat));
    }

    if (colorByStationFlag && Array.isArray(stationCodes) && stationCodes.length === origins.length) {
      // Color by station; group by station code
      const unique = [...new Set(stationCodes)];
      const palette = [0xfbbf24, 0x3b82f6, 0x22c55e, 0xec4899, 0x8b5cf6, 0x14b8a6, 0x84cc16];
      for (let s = 0; s < unique.length; s++) {
        const code = unique[s];
        const sel = stationCodes.map((c) => c === code);
        addSubset(sel, palette[s % palette.length]);
      }
    } else if (maskHits && (cbHits.checked || cbMisses.checked)) {
      if (cbHits.checked) addSubset(maskHits, 0x22c55e);
      if (cbMisses.checked) addSubset(maskMisses, 0xef4444);
    } else {
      addSubset(null, 0xfbbf24);
    }
  }

  function boxWireframeVertices(mn, mx) {
    const [x0, y0, z0] = mn;
    const [x1, y1, z1] = mx;
    const v = [
      [x0, y0, z0], [x1, y0, z0],
      [x1, y0, z0], [x1, y1, z0],
      [x1, y1, z0], [x0, y1, z0],
      [x0, y1, z0], [x0, y0, z0],

      [x0, y0, z1], [x1, y0, z1],
      [x1, y0, z1], [x1, y1, z1],
      [x1, y1, z1], [x0, y1, z1],
      [x0, y1, z1], [x0, y0, z1],

      [x0, y0, z0], [x0, y0, z1],
      [x1, y0, z0], [x1, y0, z1],
      [x1, y1, z0], [x1, y1, z1],
      [x0, y1, z0], [x0, y1, z1],
    ];
    return v.flat();
  }

  // Safari fix: create attribute using BufferAttribute constructor if missing
  function Float32BufferAttributeSafe(array, itemSize) {
    if (THREE.Float32BufferAttribute) return new THREE.Float32BufferAttribute(array, itemSize);
    return new THREE.BufferAttribute(array, itemSize);
  }

  function render() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  function loadJSONObj(obj) {
    data = obj;
    // Build layers
    buildSegments(data.segments_endpoints_f32);
    buildBoxes(data.nodes_min_f32, data.nodes_max_f32);
    maxDepth = 0;
    if (data.node_depth_i32 && data.node_depth_i32.length > 0) {
      maxDepth = data.node_depth_i32.reduce((a, b) => Math.max(a, b), 0);
    }
    depthSlider.max = String(maxDepth);
    depthSlider.value = String(maxDepth);
    depthValue.textContent = String(maxDepth);
    filterBoxesByDepth(data.node_depth_i32, maxDepth);
    const hitMask = Array.isArray(data.rays_hit_mask) ? data.rays_hit_mask : (data.rays_hit_mask || null);
    buildRays(
      data.rays_origins_f32,
      data.rays_dirs_f32,
      getRayLengthAU(),
      hitMask,
      data.rays_station_codes || null,
      colorByStation.checked,
    );
    fitToBoundsSphere(data.bounds_sphere_f32);
    // Visibility
    groupSegments.visible = cbSegments.checked;
    groupBoxes.visible = cbBoxes.checked;
    groupRays.visible = cbRays.checked;
  }

  // File input
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const obj = JSON.parse(text);
    loadJSONObj(obj);
  });

  // Toggles
  cbSegments.addEventListener('change', () => { groupSegments.visible = cbSegments.checked; });
  cbBoxes.addEventListener('change', () => { groupBoxes.visible = cbBoxes.checked; });
  cbRays.addEventListener('change', () => { groupRays.visible = cbRays.checked; });
  cbHits.addEventListener('change', () => { if (data) updateRaysOnly(); });
  cbMisses.addEventListener('change', () => { if (data) updateRaysOnly(); });
  colorByStation.addEventListener('change', () => { if (data) updateRaysOnly(); });

  depthSlider.addEventListener('input', () => {
    const d = parseInt(depthSlider.value, 10) || 0;
    depthValue.textContent = String(d);
    if (data && data.node_depth_i32) filterBoxesByDepth(data.node_depth_i32, d);
  });

  rayLen.addEventListener('input', () => { if (data) updateRaysOnly(); });

  resetViewBtn.addEventListener('click', () => {
    if (data) fitToBoundsSphere(data.bounds_sphere_f32);
  });

  // Resize
  function onResize() {
    renderer.setSize(container.clientWidth, container.clientHeight);
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', onResize);
})();


