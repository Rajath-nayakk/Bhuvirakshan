/**
 * Forest & River Guardian - Frontend Controller
 * Integrates Leaflet geospatial mapping, real-time JWT authentication,
 * priority patrol ranking, multi-sensor evidence inspection, and ranger field feedback.
 */

const API_BASE = window.location.origin;

// State management
const state = {
  token: localStorage.getItem('guardian_token') || null,
  user: JSON.parse(localStorage.getItem('guardian_user') || 'null'),
  map: null,
  layers: {
    satellite: null,
    dark: null,
    riskGrid: null,
    changePolygons: null,
    acoustic: null,
    camera: null,
  },
  alerts: [],
  selectedCellId: null,
  selectedAlertId: null,
};

document.addEventListener('DOMContentLoaded', () => {
  initAuth();
  initMap();
  setupEventListeners();
});

// --- 1. Authentication & Session Management ---
function initAuth() {
  const loginModal = document.getElementById('loginModal');
  const userBadge = document.getElementById('userBadge');
  const userNameLabel = document.getElementById('userNameLabel');
  const userRoleLabel = document.getElementById('userRoleLabel');

  if (!state.token || !state.user) {
    loginModal.classList.remove('hidden');
  } else {
    loginModal.classList.add('hidden');
    userNameLabel.textContent = state.user.full_name || state.user.email;
    userRoleLabel.textContent = state.user.role === 'admin' ? 'HQ Administrator' : 'Field Patrol Ranger';
    loadDashboardData();
  }

  // Handle Login Form Submission
  const loginForm = document.getElementById('loginForm');
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('loginEmailInput').value.trim();
    const password = document.getElementById('loginPasswordInput').value;
    const errorMsg = document.getElementById('loginErrorMsg');
    errorMsg.textContent = '';

    try {
      const resp = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!resp.ok) {
        throw new Error('Invalid email or passkey credential.');
      }

      const data = await resp.json();
      state.token = data.access_token;
      state.user = {
        email: data.user_email,
        role: data.role,
        full_name: data.full_name,
      };

      localStorage.setItem('guardian_token', state.token);
      localStorage.setItem('guardian_user', JSON.stringify(state.user));

      loginModal.classList.add('hidden');
      userNameLabel.textContent = state.user.full_name || state.user.email;
      userRoleLabel.textContent = state.user.role === 'admin' ? 'HQ Administrator' : 'Field Patrol Ranger';

      loadDashboardData();
    } catch (err) {
      errorMsg.textContent = err.message || 'Authentication failed.';
    }
  });

  // Handle Logout
  document.getElementById('btnLogoutBtn').addEventListener('click', () => {
    localStorage.removeItem('guardian_token');
    localStorage.removeItem('guardian_user');
    state.token = null;
    state.user = null;
    window.location.reload();
  });
}

function getAuthHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${state.token}`,
  };
}

// --- 2. Leaflet Map Initialization ---
function initMap() {
  // Cauvery Wildlife Sanctuary & River Corridor AOI center
  state.map = L.map('mapContainer', {
    center: [12.085, 77.225],
    zoom: 13,
    zoomControl: false,
  });

  L.control.zoom({ position: 'bottomright' }).addTo(state.map);

  // High-resolution Satellite Basemap (Esri World Imagery)
  state.layers.satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye',
      maxZoom: 18,
    }
  ).addTo(state.map);

  // CartoDB Dark Matter Alternative
  state.layers.dark = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19,
    }
  );

  // Layer Groups for Dynamic Overlays
  state.layers.riskGrid = L.layerGroup().addTo(state.map);
  state.layers.changePolygons = L.layerGroup().addTo(state.map);
  state.layers.acoustic = L.layerGroup().addTo(state.map);
  state.layers.camera = L.layerGroup().addTo(state.map);
}

// --- 3. Dashboard Data Fetching ---
async function loadDashboardData() {
  try {
    await Promise.all([
      fetchTopAlerts(),
      fetchRiskGridGeoJSON(),
      fetchChangePolygonsGeoJSON(),
      fetchDetectionsAndSensors(),
    ]);
  } catch (err) {
    console.error('Error loading dashboard feeds:', err);
  }
}

// Fetch Daily Ranked Alerts
async function fetchTopAlerts() {
  const container = document.getElementById('alertsListContainer');
  const filterVal = document.getElementById('threatFilterSelect').value;
  
  let url = `${API_BASE}/api/v1/alerts/top?limit=10`;
  if (filterVal !== 'ALL') {
    url += `&threat_level=${filterVal}`;
  }

  const resp = await fetch(url, { headers: getAuthHeaders() });
  if (!resp.ok) return;

  const data = await resp.json();
  state.alerts = data.alerts || [];

  document.getElementById('statAlertsCount').textContent = state.alerts.length;

  container.innerHTML = '';
  if (state.alerts.length === 0) {
    container.innerHTML = '<p class="subtext" style="padding: 20px; text-align: center;">No active patrol alerts matching filter.</p>';
    return;
  }

  state.alerts.forEach((alert) => {
    const card = document.createElement('div');
    const threatLower = alert.threat_level.toLowerCase();
    card.className = `alert-card ${threatLower}`;
    card.id = `alertCard_${alert.cell_id}`;

    let statusPill = '';
    if (alert.status === 'confirmed_true') {
      statusPill = '<span class="alert-status-pill confirmed_true">✓ Confirmed True</span>';
    } else if (alert.status === 'false_alarm') {
      statusPill = '<span class="alert-status-pill false_alarm">✗ False Alarm</span>';
    } else {
      statusPill = '<span class="alert-status-pill">Pending Patrol</span>';
    }

    card.innerHTML = `
      <div class="alert-card-top">
        <span class="alert-rank">#${alert.daily_rank} Priority</span>
        <span class="alert-badge badge-${threatLower}">${alert.threat_level}</span>
      </div>
      <div class="alert-cell-id">${alert.cell_id}</div>
      <div class="alert-explanation">${alert.explanation}</div>
      <div class="alert-card-footer">
        <span>Risk: <strong>${alert.risk_score.toFixed(2)}</strong></span>
        ${statusPill}
      </div>
    `;

    card.addEventListener('click', () => {
      selectCell(alert.cell_id, alert.id);
    });

    container.appendChild(card);
  });
}

// Fetch 500m Risk Grid GeoJSON
async function fetchRiskGridGeoJSON() {
  const resp = await fetch(`${API_BASE}/api/v1/grid/geojson`, { headers: getAuthHeaders() });
  if (!resp.ok) return;
  const geojson = await resp.json();

  state.layers.riskGrid.clearLayers();

  const getRiskColor = (threat) => {
    switch (threat) {
      case 'CRITICAL': return '#ef4444';
      case 'HIGH': return '#f59e0b';
      case 'MEDIUM': return '#06b6d4';
      default: return '#10b981';
    }
  };

  L.geoJSON(geojson, {
    style: (feature) => {
      const threat = feature.properties.threat_level || 'LOW';
      const color = getRiskColor(threat);
      return {
        fillColor: color,
        weight: 1,
        opacity: 0.6,
        color: '#ffffff',
        dashArray: '2',
        fillOpacity: threat === 'CRITICAL' ? 0.45 : threat === 'HIGH' ? 0.35 : 0.2,
      };
    },
    onEachFeature: (feature, layer) => {
      const props = feature.properties;
      layer.on({
        mouseover: (e) => {
          layer.setStyle({ weight: 2.5, color: '#10b981', fillOpacity: 0.55 });
        },
        mouseout: (e) => {
          layer.setStyle({
            weight: 1,
            color: '#ffffff',
            fillOpacity: props.threat_level === 'CRITICAL' ? 0.45 : props.threat_level === 'HIGH' ? 0.35 : 0.2,
          });
        },
        click: () => {
          selectCell(props.cell_id);
        },
      });

      layer.bindTooltip(`
        <strong>${props.cell_id}</strong><br/>
        Threat: <span style="color:${getRiskColor(props.threat_level)}">${props.threat_level}</span><br/>
        Risk Score: ${props.risk_score ? props.risk_score.toFixed(2) : 'N/A'}
      `, { sticky: true, className: 'glassmorphism-tooltip' });
    },
  }).addTo(state.layers.riskGrid);
}

// Fetch Satellite Change Polygons
async function fetchChangePolygonsGeoJSON() {
  const resp = await fetch(`${API_BASE}/api/v1/change/geojson`, { headers: getAuthHeaders() });
  if (!resp.ok) return;
  const geojson = await resp.json();

  state.layers.changePolygons.clearLayers();
  document.getElementById('statChangesCount').textContent = geojson.features.length;

  L.geoJSON(geojson, {
    style: (feature) => {
      const cls = feature.properties.class_name;
      let color = '#ef4444'; // sand mining red
      if (cls === 'forest_clearing') color = '#f97316';
      if (cls === 'new_track') color = '#eab308';

      return {
        fillColor: color,
        weight: 2,
        color: '#ffffff',
        fillOpacity: 0.6,
      };
    },
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      layer.bindPopup(`
        <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
          <h4 style="margin: 0 0 4px 0; color: #dc2626;">Satellite Change Detection</h4>
          <p><strong>Class:</strong> ${p.class_name || 'Sand Mining'}</p>
          <p><strong>Area:</strong> ${p.area_ha ? p.area_ha.toFixed(2) : 1.44} ha</p>
          <p><strong>Confidence:</strong> ${(p.confidence * 100).toFixed(0)}%</p>
        </div>
      `);
    },
  }).addTo(state.layers.changePolygons);
}

// Fetch Sensor Detections & Markers
async function fetchDetectionsAndSensors() {
  const resp = await fetch(`${API_BASE}/api/v1/detections?limit=100`, { headers: getAuthHeaders() });
  if (!resp.ok) return;
  const detections = await resp.json();

  state.layers.acoustic.clearLayers();
  state.layers.camera.clearLayers();

  let acousticCount = 0;
  let cameraCount = 0;

  detections.forEach((d) => {
    if (!d.coordinates) return; // Obfuscated or private
    const lat = d.coordinates[1];
    const lon = d.coordinates[0];

    if (d.source === 'acoustic') {
      acousticCount++;
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        fillColor: '#f59e0b',
        color: '#ffffff',
        weight: 1.5,
        fillOpacity: 0.9,
      });

      marker.bindPopup(`
        <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
          <h4 style="margin:0 0 4px 0; color:#d97706;">Acoustic Intercept</h4>
          <p><strong>Type:</strong> ${d.class_name}</p>
          <p><strong>Confidence:</strong> ${(d.confidence * 100).toFixed(0)}%</p>
          <p><strong>Time:</strong> ${new Date(d.timestamp).toLocaleTimeString()}</p>
        </div>
      `);
      marker.addTo(state.layers.acoustic);
    } else if (d.source === 'camera') {
      cameraCount++;
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        fillColor: '#06b6d4',
        color: '#ffffff',
        weight: 1.5,
        fillOpacity: 0.9,
      });

      marker.bindPopup(`
        <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
          <h4 style="margin:0 0 4px 0; color:#0891b2;">Camera Trap Alert</h4>
          <p><strong>Object:</strong> ${d.class_name}</p>
          <p><strong>Confidence:</strong> ${(d.confidence * 100).toFixed(0)}%</p>
          <p><strong>Curfew Threat:</strong> ${d.threat_level}</p>
        </div>
      `);
      marker.addTo(state.layers.camera);
    }
  });

  document.getElementById('statAcousticCount').textContent = acousticCount;
  document.getElementById('statCameraCount').textContent = cameraCount;
}

// --- 4. Cell Detail & Forensic Evidence Drawer ---
async function selectCell(cellId, alertId = null) {
  state.selectedCellId = cellId;
  state.selectedAlertId = alertId;

  // Highlight card in sidebar
  document.querySelectorAll('.alert-card').forEach((c) => c.classList.remove('active'));
  const card = document.getElementById(`alertCard_${cellId}`);
  if (card) {
    card.classList.add('active');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Open drawer
  const drawer = document.getElementById('detailDrawer');
  drawer.classList.add('open');

  document.getElementById('drawerCellId').textContent = cellId;
  document.getElementById('feedbackStatusMsg').textContent = '';
  document.getElementById('feedbackNotesInput').value = '';

  try {
    const resp = await fetch(`${API_BASE}/api/v1/cells/${cellId}`, { headers: getAuthHeaders() });
    if (!resp.ok) return;
    const detail = await resp.json();

    document.getElementById('drawerRiskScore').textContent = detail.risk_score.toFixed(2);
    document.getElementById('drawerExplanation').textContent = detail.explanation;
    
    const badge = document.getElementById('drawerThreatBadge');
    badge.textContent = detail.threat_level;
    badge.className = `badge badge-${detail.threat_level.toLowerCase()}`;

    // Pan map to cell centroid
    if (detail.centroid_lat && detail.centroid_lon) {
      state.map.flyTo([detail.centroid_lat, detail.centroid_lon], 14, { duration: 0.8 });
    }

    // Set alert ID from associated alerts if not provided
    if (!state.selectedAlertId && detail.associated_alerts && detail.associated_alerts.length > 0) {
      state.selectedAlertId = detail.associated_alerts[0].id;
    }
  } catch (err) {
    console.error('Error loading cell details:', err);
  }
}

// --- 5. Event Listeners & UI Controls ---
function setupEventListeners() {
  // Close drawer button
  document.getElementById('btnCloseDrawer').addEventListener('click', () => {
    document.getElementById('detailDrawer').classList.remove('open');
  });

  // Threat filter change
  document.getElementById('threatFilterSelect').addEventListener('change', () => {
    fetchTopAlerts();
  });

  document.getElementById('btnRefreshAlerts').addEventListener('click', () => {
    fetchTopAlerts();
  });

  // Evidence Tabs switching
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach((tc) => tc.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
    });
  });

  // Map Layer Toggles
  document.getElementById('toggleRiskGrid').addEventListener('change', (e) => {
    if (e.target.checked) state.map.addLayer(state.layers.riskGrid);
    else state.map.removeLayer(state.layers.riskGrid);
  });

  document.getElementById('toggleChangePolygons').addEventListener('change', (e) => {
    if (e.target.checked) state.map.addLayer(state.layers.changePolygons);
    else state.map.removeLayer(state.layers.changePolygons);
  });

  document.getElementById('toggleAcousticSensors').addEventListener('change', (e) => {
    if (e.target.checked) state.map.addLayer(state.layers.acoustic);
    else state.map.removeLayer(state.layers.acoustic);
  });

  document.getElementById('toggleCameraTraps').addEventListener('change', (e) => {
    if (e.target.checked) state.map.addLayer(state.layers.camera);
    else state.map.removeLayer(state.layers.camera);
  });

  document.getElementById('toggleBaseLayer').addEventListener('change', (e) => {
    if (e.target.checked) {
      state.map.removeLayer(state.layers.dark);
      state.map.addLayer(state.layers.satellite);
    } else {
      state.map.removeLayer(state.layers.satellite);
      state.map.addLayer(state.layers.dark);
    }
  });

  // Ranger Feedback: Confirm True Alert
  document.getElementById('btnConfirmAlert').addEventListener('click', () => {
    submitRangerFeedback(true);
  });

  // Ranger Feedback: Flag False Alarm
  document.getElementById('btnFalseAlarm').addEventListener('click', () => {
    submitRangerFeedback(false);
  });

  // Web Audio Tone Intercept Simulator
  document.getElementById('btnPlayAudioSim').addEventListener('click', () => {
    playAcousticPulse();
  });
}

// Submit Ranger Verification Feedback
async function submitRangerFeedback(isTrueAlert) {
  if (!state.selectedCellId) return;

  const notes = document.getElementById('feedbackNotesInput').value.trim();
  const statusMsg = document.getElementById('feedbackStatusMsg');
  statusMsg.textContent = 'Submitting field report...';
  statusMsg.className = 'status-msg';

  try {
    const payload = {
      alert_id: state.selectedAlertId || null,
      cell_id: state.selectedCellId,
      is_true_alert: isTrueAlert,
      notes: notes || (isTrueAlert ? 'Field confirmed illegal extraction.' : 'False alarm - legal activity.'),
    };

    const resp = await fetch(`${API_BASE}/api/v1/feedback`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error('Feedback submission failed.');

    const data = await resp.json();
    statusMsg.textContent = isTrueAlert ? '✓ Confirmed as True Alert. Enforcement notified!' : '✓ Alert dismissed as False Alarm.';
    statusMsg.className = 'status-msg success';

    // Refresh alert list to show updated status pill
    fetchTopAlerts();
  } catch (err) {
    statusMsg.textContent = err.message || 'Error submitting verification.';
    statusMsg.className = 'status-msg error';
  }
}

// Synthesizer Web Audio Beep/Waveform Simulator
function playAcousticPulse() {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(220, audioCtx.currentTime); // Low gunshot thump
    osc.frequency.exponentialRampToValueAtTime(40, audioCtx.currentTime + 0.3);

    gain.gain.setValueAtTime(0.8, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.36);
  } catch (e) {
    console.log('Web Audio tone simulation active.');
  }
}
