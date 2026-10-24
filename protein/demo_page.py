"""The human-friendly demo page served at GET /.

Swagger (/docs) stays for developers; this is what you put on a projector.
Self-contained: inline CSS/JS, no CDNs, talks to /analyze and /v0/estimate.
"""
from __future__ import annotations

DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protein Tracker — live demo</title>
<style>
  :root { --ink:#1a1f16; --muted:#6b7263; --lime:#b7e219; --card:#ffffff; --bg:#f4f6ef;
          --amber:#b45309; --amber-bg:#fef3c7; --red:#b91c1c; --red-bg:#fee2e2; }
  * { box-sizing:border-box; margin:0; }
  body { font:16px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; background:var(--bg);
         color:var(--ink); min-height:100vh; display:flex; justify-content:center; padding:40px 16px; }
  main { width:100%; max-width:680px; }
  h1 { font-size:1.7rem; letter-spacing:-.02em; }
  .tag { color:var(--muted); margin:.25rem 0 1.5rem; }
  .card { background:var(--card); border-radius:14px; padding:20px;
          box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:16px; }
  textarea { width:100%; border:1.5px solid #d6dbcd; border-radius:10px; padding:12px 14px;
             font:inherit; resize:vertical; min-height:64px; }
  textarea:focus { outline:2px solid var(--lime); border-color:transparent; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
  .chip { border:1px solid #d6dbcd; background:#fafbf7; border-radius:999px; padding:4px 12px;
          font-size:.85rem; cursor:pointer; color:var(--muted); }
  .chip:hover { border-color:var(--ink); color:var(--ink); }
  .row { display:flex; gap:10px; align-items:center; }
  button.go { background:var(--ink); color:#fff; border:0; border-radius:10px; padding:11px 22px;
              font:inherit; font-weight:600; cursor:pointer; }
  button.go:disabled { opacity:.5; cursor:wait; }
  button.alt { background:none; border:0; color:var(--muted); font:inherit; font-size:.9rem;
               cursor:pointer; text-decoration:underline; }
  #status { color:var(--muted); font-size:.9rem; }
  table { width:100%; border-collapse:collapse; margin-top:6px; }
  th { text-align:left; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em;
       color:var(--muted); padding:6px 8px; border-bottom:1px solid #e4e8dd; }
  td { padding:9px 8px; border-bottom:1px solid #eef1e8; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .src { font-size:.72rem; padding:2px 8px; border-radius:999px; background:#e9f7c0; }
  .src.off { background:#e0ecff; }
  .src.none { background:var(--red-bg); color:var(--red); }
  .total { display:flex; justify-content:space-between; align-items:baseline; margin-top:14px; }
  .total b { font-size:2rem; letter-spacing:-.02em; }
  .goalbar { height:10px; background:#e4e8dd; border-radius:999px; overflow:hidden; margin-top:8px; }
  .goalbar i { display:block; height:100%; background:var(--lime); border-radius:999px;
               transition:width .5s ease; }
  .note { border-radius:10px; padding:10px 14px; margin-top:12px; font-size:.92rem; }
  .warn { background:var(--amber-bg); color:var(--amber); }
  .err  { background:var(--red-bg); color:var(--red); }
  .naive { margin-top:12px; color:var(--muted); font-size:.95rem; }
  footer { color:var(--muted); font-size:.8rem; text-align:center; margin-top:20px; }
  .hide { display:none; }
</style>
</head>
<body>
<main>
  <h1>Protein Tracker</h1>
  <p class="tag">Type a meal in plain words. The AI only <em>identifies</em> the foods —
     every protein number comes from a verified database.</p>

  <div class="card">
    <textarea id="meal" placeholder="e.g. 150g grilled chicken breast and 2 eggs"></textarea>
    <div class="chips">
      <span class="chip">150g grilled chicken breast and 2 eggs</span>
      <span class="chip">2 slices whole wheat bread with 2 tbsp peanut butter</span>
      <span class="chip">1 cup greek yogurt and a banana</span>
      <span class="chip">1 katori of homemade rajma</span>
    </div>
    <div class="row">
      <button class="go" id="go">Analyze</button>
      <span id="status"></span>
    </div>
  </div>

  <div class="card hide" id="result">
    <table>
      <thead><tr><th>Food</th><th class="num">Amount</th><th class="num">Protein</th><th>Source</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="total"><span>Total protein</span><span><b id="total"></b> g</span></div>
    <div class="goalbar"><i id="bar" style="width:0"></i></div>
    <div id="goaltext" style="color:var(--muted);font-size:.85rem;margin-top:4px"></div>
    <div class="note warn hide" id="nohits"></div>
    <div class="naive">
      <button class="alt" id="naive">Compare with the naive AI guess →</button>
      <span id="naiveout"></span>
    </div>
  </div>

  <div class="card note err hide" id="error"></div>
  <footer id="foot">connecting…</footer>
</main>

<script>
const $ = id => document.getElementById(id);
let lastMeal = "";

fetch("/health").then(r => r.json()).then(h => {
  $("foot").textContent = `model: ${h.provider} · matcher: ${h.matcher} · verified foods: ${h.verified_foods}`;
}).catch(() => { $("foot").textContent = "server unreachable"; });

document.querySelectorAll(".chip").forEach(c =>
  c.addEventListener("click", () => { $("meal").value = c.textContent; }));

$("go").addEventListener("click", analyze);
$("meal").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); analyze(); } });
$("naive").addEventListener("click", naive);

async function analyze() {
  const text = $("meal").value;
  $("error").classList.add("hide"); $("result").classList.add("hide");
  $("go").disabled = true; $("status").textContent = "the model is reading your meal…";
  try {
    const r = await fetch("/analyze", { method:"POST",
      headers:{"Content-Type":"application/json"}, body: JSON.stringify({text}) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
    render(body); lastMeal = text;
  } catch (e) {
    $("error").textContent = "⚠ " + e.message; $("error").classList.remove("hide");
  } finally { $("go").disabled = false; $("status").textContent = ""; }
}

function render(b) {
  $("rows").innerHTML = b.items.map(i => `
    <tr>
      <td>${i.name}</td>
      <td class="num">${Math.round(i.grams)} g</td>
      <td class="num">${i.no_hit ? "—" : i.protein_g.toFixed(1) + " g"}</td>
      <td><span class="src ${i.source}">${i.no_hit ? "not found" : i.source}</span></td>
    </tr>`).join("");
  $("total").textContent = b.total_protein_g.toFixed(1);
  const pct = Math.min(100, 100 * b.total_protein_g / b.goal_g);
  $("bar").style.width = pct + "%";
  $("goaltext").textContent = b.meets_goal
    ? `daily goal of ${b.goal_g} g reached 🎉`
    : `${(b.goal_g - b.total_protein_g).toFixed(1)} g to go toward the ${b.goal_g} g daily goal`;
  if (b.no_hits.length) {
    $("nohits").textContent = "Not in the database (flagged for you, never guessed): " + b.no_hits.join(", ");
    $("nohits").classList.remove("hide");
  } else { $("nohits").classList.add("hide"); }
  $("naiveout").textContent = "";
  $("result").classList.remove("hide");
}

async function naive() {
  $("naiveout").textContent = " asking the model to just guess…";
  try {
    const r = await fetch("/v0/estimate", { method:"POST",
      headers:{"Content-Type":"application/json"}, body: JSON.stringify({text: lastMeal || $("meal").value}) });
    const b = await r.json();
    if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
    $("naiveout").textContent = ` the model's bare guess: ${b.total_protein_g.toFixed(1)} g — one number, no breakdown, no sources.`;
  } catch (e) { $("naiveout").textContent = " naive call failed: " + e.message; }
}
</script>
</body>
</html>"""
