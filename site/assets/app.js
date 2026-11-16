// shared: nav, theme, helpers, charts.  window.DATA is defined in data.js
const $ = id => document.getElementById(id);
const fmt = (x, d = 1) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PAGES = [["index.html", "Today"], ["map.html", "City map"], ["models.html", "Model showdown"], ["results.html", "Real-station results"], ["reality.html", "Model data vs reality"], ["method.html", "How it works"]];
function nav() {
  const here = location.pathname.split("/").pop() || "index.html";
  const n = document.createElement("div"); n.className = "nav";
  n.innerHTML = `<div class="in"><div class="brand"><span class="dot"></span>Almaty Smog Desk</div><div class="links">${PAGES.map(([h, t]) => `<a href="${h}" class="${h === here ? "on" : ""}">${t}</a>`).join("")}</div><span class="snap" id="snap"></span><button class="theme" id="theme" title="Toggle theme">◐</button></div>`;
  document.body.prepend(n);
  try { const t = new Date(DATA.snapshot_utc), m = Math.round((Date.now() - t) / 60000);
    const ago = m < 60 ? m + " min ago" : m < 1440 ? Math.round(m / 60) + " h ago" : Math.round(m / 1440) + " d ago";
    $("snap").innerHTML = `<i></i>Live snapshot · ${ago}`; $("snap").title = "Data refreshed automatically every hour. Snapshot: " + t.toLocaleString(); } catch (e) {}
  const root = document.documentElement; let t; try { t = localStorage.getItem("theme"); } catch (e) {}
  if (t) root.dataset.theme = t;
  $("theme").onclick = () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
    redraw();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
}
const charts = [], builders = [];
var redraw;
function chart(id, build) { builders.push([id, build]); }
redraw = function () { charts.splice(0).forEach(c => c.destroy()); builders.forEach(([id, b]) => charts.push(new Chart($(id), b()))); }
function baseOpts(ytitle, labels) {
  return { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: i => (labels ? labels[i.datasetIndex] + ": " : "") + fmt(i.raw) } } },
    scales: { x: { ticks: { color: css("--muted"), maxTicksLimit: 8, font: { family: "Inter" } }, grid: { display: false } },
              y: { beginAtZero: true, ticks: { color: css("--muted"), font: { family: "Inter" } }, grid: { color: css("--line") },
                   title: { display: !!ytitle, text: ytitle, color: css("--muted") } } } };
}
const line = (data, color, w = 1.5, dash) => ({ data, borderColor: color, backgroundColor: color, borderWidth: w, pointRadius: 0, tension: .25, borderDash: dash || [], spanGaps: false });
const limitLine = (n, v) => line(Array(n).fill(v), css("--red"), 1, [2, 4]);
const table = (head, body) => `<thead><tr>${head.map(h => `<th class="${h[1] || ""}">${h[0]}</th>`).join("")}</tr></thead><tbody>${body.join("")}</tbody>`;
const legend = items => items.map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join("");
const dayName = d => new Date(d + "T12:00").toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
document.addEventListener("DOMContentLoaded", () => { nav(); redraw(); });
