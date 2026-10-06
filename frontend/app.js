const statusBox = document.querySelector("#status");
const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const prevStepButton = document.querySelector("#prev-step");
const nextStepButton = document.querySelector("#next-step");
const predictButton = document.querySelector("#predict-button");
let currentStep = 1;
const API_BASE = window.location.origin.startsWith("http")
  && ["8000", "8001"].includes(window.location.port)
  ? window.location.origin
  : "http://127.0.0.1:8001";

const defaults = {
  hotel: "City Hotel",
  arrival_date_month: "January",
  meal: "BB",
  country: "PRT",
  market_segment: "Online TA",
  distribution_channel: "TA/TO",
  reserved_room_type: "A",
  deposit_type: "No Deposit",
  customer_type: "Transient",
  agent: "9",
  company: "Not_recorded",
};
const MONTH_ORDER = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

function fillSelect(name, values) {
  const select = form.elements[name];
  if (!select) return;
  if (name === "arrival_date_month") {
    values = MONTH_ORDER.filter((month) => values.includes(month));
  }
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
  if (defaults[name] && values.includes(defaults[name])) select.value = defaults[name];
}

function setStatus(ok, text) {
  statusBox.textContent = text;
  statusBox.classList.toggle("online", ok);
  statusBox.classList.toggle("offline", !ok);
}

function number(value) {
  return new Intl.NumberFormat().format(value);
}

function renderKpis(overview) {
  const items = [
    ["Bookings reviewed", number(overview.total_bookings)],
    ["Overall cancellation risk", `${overview.cancellation_rate}%`],
    ["Average days booked early", `${overview.avg_lead_time} days`],
    ["Average room rate", overview.avg_adr],
    ["Average stay length", `${overview.avg_stay} nights`],
    ["Booking details used", overview.predictor_count],
    ["Cancelled bookings", number(overview.cancelled_bookings)],
    ["Completed bookings", number(overview.not_cancelled_bookings)],
  ];
  document.querySelector("#kpi-grid").innerHTML = items
    .map(([label, value]) => `<article class="card kpi"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
}

function renderDonut(id, rows) {
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const cancelled = rows.find((row) => row.label === "Cancelled")?.count || 0;
  const rate = total ? Math.round((cancelled / total) * 1000) / 10 : 0;
  document.querySelector(`#${id}`).innerHTML = `
    <div class="donut" style="--value: ${rate}">
      <span>${rate}%</span>
    </div>
    <div class="legend-list">
      ${rows.map((row) => `
        <div><span>${row.label}</span><strong>${number(row.count)}</strong></div>
      `).join("")}
    </div>
  `;
}

function renderPie(id, rows) {
  const total = rows.reduce((sum, row) => sum + row.bookings, 0) || 1;
  const colors = ["#35c9ff", "#ffb454", "#4ade80", "#ff5c77", "#a78bfa", "#f472b6"];
  let cumulative = 0;
  const segments = rows.map((row, index) => {
    const start = cumulative / total;
    cumulative += row.bookings;
    const end = cumulative / total;
    return `${colors[index % colors.length]} ${start * 100}% ${end * 100}%`;
  }).join(", ");
  document.querySelector(`#${id}`).innerHTML = `
    <div class="pie" style="background: conic-gradient(${segments})"></div>
    <div class="legend-list compact">
      ${rows.map((row, index) => `
        <div><span><i style="background:${colors[index % colors.length]}"></i>${row.label}</span><strong>${row.cancellation_rate}%</strong></div>
      `).join("")}
    </div>
  `;
}

function renderColumnChart(id, rows) {
  const valueKey = rows.some((row) => row.count !== undefined) ? "count" : "cancellation_rate";
  const max = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);
  document.querySelector(`#${id}`).innerHTML = `
    <div class="columns">
      ${rows.map((row) => {
        const value = Number(row[valueKey]) || 0;
        const height = Math.max((value / max) * 100, 8);
        const suffix = valueKey === "cancellation_rate" ? "%" : "";
        return `
          <div class="column-item">
            <div class="column-value">${valueKey === "count" ? number(value) : value}${suffix}</div>
            <div class="column-track"><span style="height:${height}%"></span></div>
            <small>${row.label}</small>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderLineChart(id, rows) {
  const orderedMonths = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ];
  const sorted = [...rows].sort((a, b) => orderedMonths.indexOf(a.label) - orderedMonths.indexOf(b.label));
  const values = sorted.map((row) => Number(row.cancellation_rate) || 0);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = Math.max(max - min, 1);
  const points = values.map((value, index) => {
    const x = 16 + (index * (268 / Math.max(values.length - 1, 1)));
    const y = 128 - ((value - min) / span) * 96;
    return `${x},${y}`;
  }).join(" ");
  document.querySelector(`#${id}`).innerHTML = `
    <svg class="sparkline" viewBox="0 0 300 150" aria-label="Monthly cancellation rate line chart">
      <polyline points="${points}" fill="none" stroke="#ffb454" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      ${points.split(" ").map((point) => `<circle cx="${point.split(",")[0]}" cy="${point.split(",")[1]}" r="4" fill="#35c9ff"/>`).join("")}
    </svg>
    <div class="line-summary">
      <span>Lowest: ${Math.round(min * 10) / 10}%</span>
      <strong>Highest: ${Math.round(max * 10) / 10}%</strong>
    </div>
  `;
}

function renderBubbleChart(id, rows) {
  const maxBookings = Math.max(...rows.map((row) => row.bookings), 1);
  const maxRisk = Math.max(...rows.map((row) => row.cancellation_rate), 1);
  document.querySelector(`#${id}`).innerHTML = `
    <div class="bubble-stage">
      ${rows.slice(0, 7).map((row, index) => {
        const size = 34 + (row.bookings / maxBookings) * 58;
        const left = 8 + (index % 4) * 23;
        const top = 10 + Math.floor(index / 4) * 42 + (maxRisk - row.cancellation_rate) * 0.2;
        return `
          <div class="bubble" style="width:${size}px;height:${size}px;left:${left}%;top:${top}%">
            <strong>${row.cancellation_rate}%</strong>
            <span>${row.label}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderBars(id, rows, valueKey, label, color = "cyan") {
  const max = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);
  document.querySelector(`#${id}`).innerHTML = rows.map((row) => {
    const value = Number(row[valueKey]) || 0;
    const width = Math.max((value / max) * 100, 3);
    const suffix = valueKey.includes("rate") ? "%" : "";
    return `
      <div class="bar-row ${color}">
        <div class="bar-meta">
          <span title="${row.label}">${row.label}</span>
          <strong>${value}${suffix}</strong>
        </div>
        <div class="bar-track"><span style="width: ${width}%"></span></div>
        ${row.bookings ? `<small>${number(row.bookings)} bookings</small>` : `<small>${label}</small>`}
      </div>
    `;
  }).join("");
}

function renderComparisonCards(id, rows) {
  const total = rows.reduce((sum, row) => sum + row.count, 0) || 1;
  document.querySelector(`#${id}`).innerHTML = rows.map((row) => {
    const share = Math.round((row.count / total) * 1000) / 10;
    return `
      <div class="compare-card">
        <span>${row.label}</span>
        <strong>${number(row.count)}</strong>
        <div class="mini-track"><span style="width: ${share}%"></span></div>
        <small>${share}% of all bookings</small>
      </div>
    `;
  }).join("");
}

function renderRiskGrid(id, rows) {
  document.querySelector(`#${id}`).innerHTML = rows.map((row) => {
    const risk = Number(row.cancellation_rate) || 0;
    const level = risk >= 50 ? "high" : risk >= 30 ? "medium" : "low";
    return `
      <div class="risk-tile ${level}">
        <span>${row.label}</span>
        <strong>${risk}%</strong>
        <small>${number(row.bookings)} bookings</small>
      </div>
    `;
  }).join("");
}

function renderDashboard(data) {
  const { overview, charts: dashboardCharts } = data;
  renderKpis(overview);

  renderDonut("targetChart", dashboardCharts.target_distribution);
  renderColumnChart("hotelChart", dashboardCharts.hotel_type);
  renderRiskGrid("customerChart", dashboardCharts.customer_type);
  renderBars("monthChart", dashboardCharts.arrival_month, "cancellation_rate", "Cancellation rate", "amber");
  renderBubbleChart("marketChart", dashboardCharts.market_segment);
  renderPie("depositChart", dashboardCharts.deposit_type);
  renderLineChart("leadTimeChart", dashboardCharts.lead_time_bands);
  renderBars("countryChart", dashboardCharts.top_countries, "cancellation_rate", "Cancellation rate", "cyan");
  renderBars("importanceChart", dashboardCharts.feature_importance, "importance", "Feature importance", "cyan");
  renderRecommendations(data.recommendations || []);
}

function renderRecommendations(items) {
  const priorityClass = { High: "high", Medium: "medium", Low: "low" };
  document.querySelector("#recommendation-list").innerHTML = items.map((item) => `
    <article class="recommendation-card ${priorityClass[item.priority] || "low"}">
      <div><strong>${item.title}</strong><span>${item.priority}</span></div>
      <p>${item.detail}</p>
    </article>
  `).join("");
}

function updateStep() {
  document.querySelectorAll(".form-step").forEach((step) => {
    step.classList.toggle("active", Number(step.dataset.step) === currentStep);
  });
  document.querySelectorAll(".step-dot").forEach((dot, index) => {
    dot.classList.toggle("active", index + 1 === currentStep);
  });
  prevStepButton.style.display = currentStep === 1 ? "none" : "inline-flex";
  nextStepButton.style.display = currentStep === 1 ? "inline-flex" : "none";
  predictButton.style.display = currentStep === 2 ? "inline-flex" : "none";
}

function currentStepValid() {
  const step = document.querySelector(`.form-step[data-step="${currentStep}"]`);
  const fields = [...step.querySelectorAll("input, select")];
  return fields.every((field) => field.reportValidity());
}

function formPayload() {
  return Object.fromEntries(new FormData(form).entries());
}

function showErrors(errors) {
  const items = Object.values(errors).map((message) => `<li>${message}</li>`).join("");
  result.innerHTML = `<div class="errors"><strong>Please check these inputs:</strong><ul>${items}</ul></div>`;
}

function showPrediction(data) {
  const percent = data.cancellation_probability === null ? null : Math.round(data.cancellation_probability * 1000) / 10;
  const className = data.prediction === 1 ? "risk" : "good";
  const advice = data.prediction === 1
    ? "Recommended action: contact the guest, confirm arrival plans, and review deposit or payment status."
    : "Recommended action: treat as lower risk, while still following normal confirmation procedures.";
  result.innerHTML = `
    <div>
      <div class="prediction ${className}">${data.prediction_label}</div>
      <p>This estimate is based on similar past hotel bookings.</p>
      ${
        percent === null
          ? ""
          : `<p><strong>Estimated cancellation chance: ${percent}%</strong></p>
             <div class="bar"><span style="width: ${percent}%"></span></div>`
      }
      <p class="muted">${advice}</p>
    </div>
  `;
}

function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach((button) => {
    button.addEventListener("click", () => {
      activateSection(button.dataset.section);
    });
  });
}

function activateSection(sectionName) {
  const button = document.querySelector(`.nav-link[data-section="${sectionName}"]`);
  const section = document.querySelector(`#section-${sectionName}`);
  if (!button || !section) return;
  document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".section").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  section.classList.add("active");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.innerHTML = `<p class="muted">Generating prediction...</p>`;
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formPayload()),
  });
  const data = await response.json();
  if (!response.ok) {
    showErrors(data.errors || { request: data.error || "Prediction request failed." });
    return;
  }
  showPrediction(data);
});

nextStepButton.addEventListener("click", () => {
  if (!currentStepValid()) return;
  currentStep = 2;
  updateStep();
});

prevStepButton.addEventListener("click", () => {
  currentStep = 1;
  updateStep();
});

form.addEventListener("reset", () => {
  window.setTimeout(() => {
    currentStep = 1;
    updateStep();
    result.innerHTML = `
      <h2>Prediction Result</h2>
      <p class="muted">The cancellation risk will appear here after you submit a booking.</p>
    `;
  }, 0);
});

async function init() {
  setupNavigation();
  activateSection(new URLSearchParams(window.location.search).get("section") || "overview");
  updateStep();
  let health;
  try {
    health = await apiGet("/api/health");
    setStatus(true, "System ready");
  } catch (error) {
    setStatus(false, "API offline");
    throw error;
  }

  const [schema, dashboard] = await Promise.all([
    apiGet("/api/schema"),
    apiGet("/api/dashboard"),
  ]);
  Object.entries(schema.options).forEach(([name, values]) => fillSelect(name, values));
  renderDashboard(dashboard);
}

init().catch((error) => {
  result.innerHTML = `<div class="errors">Could not load dashboard data: ${error.message}</div>`;
});
