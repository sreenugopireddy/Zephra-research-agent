"""A small, self-contained HTML dashboard for exercising POST /research.

Served at GET /dashboard by app/main.py. Plain HTML/CSS/vanilla JS, no build
step, no framework, no external requests — it calls the API on the same
origin it's served from, so no CORS configuration is needed.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Research Agent Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161922; --border: #262b38; --text: #e6e8ee;
    --muted: #8b93a7; --accent: #5b8cff; --green: #35c46b; --red: #ef5757;
    --yellow: #e8b93f; --gray: #5b6272;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 16px 64px; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px; margin-bottom: 20px;
  }
  textarea {
    width: 100%; min-height: 84px; resize: vertical; background: #0c0e13;
    color: var(--text); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; font-size: 14px; font-family: inherit;
  }
  .row { display: flex; gap: 16px; align-items: center; margin-top: 14px; flex-wrap: wrap; }
  .row label { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  .row input[type="number"] {
    width: 56px; background: #0c0e13; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 6px; font-size: 13px;
  }
  button {
    background: var(--accent); color: white; border: none; border-radius: 8px;
    padding: 10px 20px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 16px;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status-line { display: flex; gap: 8px; align-items: center; margin-top: 10px; font-size: 13px; color: var(--muted); }
  .spinner {
    width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.7s linear infinite; display: none;
  }
  .spinner.on { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em;
  }
  .b-success, .b-high, .b-passed, .b-completed { background: rgba(53,196,107,0.15); color: var(--green); }
  .b-failed, .b-unavailable, .b-unsupported { background: rgba(239,87,87,0.15); color: var(--red); }
  .b-partial, .b-medium { background: rgba(232,185,63,0.15); color: var(--yellow); }
  .b-skipped, .b-not_run, .b-low { background: rgba(91,98,114,0.2); color: var(--gray); }
  .answer { white-space: pre-wrap; line-height: 1.6; font-size: 14.5px; }
  .answer .cite {
    background: rgba(91,140,255,0.15); color: var(--accent); padding: 0 4px;
    border-radius: 4px; font-size: 12px; font-weight: 600;
  }
  h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .claim { border-left: 2px solid var(--border); padding: 8px 0 8px 12px; margin-bottom: 8px; font-size: 13.5px; }
  .claim .meta { margin-top: 4px; display: flex; gap: 6px; align-items: center; }
  .source { padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13.5px; }
  .source:last-child { border-bottom: none; }
  .source a { color: var(--accent); text-decoration: none; word-break: break-all; }
  .source .meta { margin-top: 4px; color: var(--muted); font-size: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  ul.plain { margin: 0; padding-left: 18px; font-size: 13.5px; }
  ul.plain li { margin-bottom: 6px; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
  .stat { background: #0c0e13; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .stat .val { font-size: 18px; font-weight: 700; }
  .stat .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; margin-top: 2px; }
  .provider-row { display: flex; gap: 10px; flex-wrap: wrap; }
  details summary { cursor: pointer; color: var(--muted); font-size: 13px; }
  pre.debug {
    background: #0c0e13; border: 1px solid var(--border); border-radius: 8px; padding: 12px;
    overflow-x: auto; font-size: 12px; margin-top: 10px;
  }
  .error-box {
    background: rgba(239,87,87,0.1); border: 1px solid var(--red); border-radius: 8px;
    padding: 12px; font-size: 13.5px; color: var(--red);
  }
  .empty { color: var(--muted); font-size: 13px; font-style: italic; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Multi-Source Web Research Agent</h1>
  <div class="sub">Ask a research question — searched, deduplicated, ranked, and answered with citations.</div>

  <div class="panel">
    <textarea id="question" placeholder="e.g. What are the trade-offs between RAG and fine-tuning for enterprise customer support?"></textarea>
    <div class="row">
      <label>Max iterations <input type="number" id="maxIterations" value="2" min="1" max="2" /></label>
      <label>Max sources <input type="number" id="maxSources" value="6" min="1" max="6" /></label>
      <label><input type="checkbox" id="includeDebug" /> Include debug info</label>
    </div>
    <button id="submitBtn" onclick="runResearch()">Research</button>
    <div class="status-line">
      <span class="spinner" id="spinner"></span>
      <span id="statusText"></span>
    </div>
  </div>

  <div id="results"></div>
</div>

<script>
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderAnswer(markdown) {
  let html = escapeHtml(markdown || "");
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
  html = html.replace(/\\[(S\\d+|unsupported)\\]/g, '<span class="cite">[$1]</span>');
  return html;
}

function badge(text, cls) {
  return '<span class="badge b-' + cls + '">' + escapeHtml(String(text)) + '</span>';
}

function renderClaims(claims) {
  if (!claims || claims.length === 0) return '<div class="empty">No individual claims returned.</div>';
  return claims.map(function (c) {
    var ids = (c.source_ids || []).map(function (id) { return badge(id, "success"); }).join(" ");
    return '<div class="claim">' + escapeHtml(c.text) +
      '<div class="meta">' + badge(c.confidence, c.confidence) + " " + ids + '</div></div>';
  }).join("");
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return '<div class="empty">No sources were selected.</div>';
  return sources.map(function (s) {
    return '<div class="source">' +
      '<strong>[' + escapeHtml(s.source_id) + ']</strong> ' + escapeHtml(s.title || s.url) +
      '<br/><a href="' + escapeHtml(s.url) + '" target="_blank" rel="noopener">' + escapeHtml(s.url) + '</a>' +
      '<div class="meta">' +
      (s.provider ? escapeHtml(s.provider) + " · " : "") +
      escapeHtml(s.domain || "") + " · " +
      badge(s.extraction_status, s.extraction_status) +
      '</div></div>';
  }).join("");
}

function renderList(items) {
  if (!items || items.length === 0) return '<div class="empty">None reported.</div>';
  return '<ul class="plain">' + items.map(function (i) { return "<li>" + escapeHtml(i) + "</li>"; }).join("") + "</ul>";
}

function renderProviderStatus(status) {
  return '<div class="provider-row">' + Object.keys(status || {}).map(function (name) {
    return '<div>' + escapeHtml(name) + ": " + badge(status[name], status[name]) + "</div>";
  }).join("") + "</div>";
}

function stat(value, label) {
  return '<div class="stat"><div class="val">' + escapeHtml(String(value)) + '</div><div class="lbl">' + escapeHtml(label) + "</div></div>";
}

function renderResults(data) {
  var m = data.metadata || {};
  var html = "";

  html += '<div class="panel"><h2>Answer</h2><div class="answer">' + renderAnswer(data.answer) + "</div></div>";

  html += '<div class="panel"><h2>Claims</h2>' + renderClaims(data.claims) + "</div>";

  html += '<div class="panel"><h2>Sources</h2>' + renderSources(data.sources) + "</div>";

  if ((data.conflicts && data.conflicts.length) || (data.uncertainties && data.uncertainties.length)) {
    html += '<div class="panel"><h2>Conflicts</h2>' + renderList(data.conflicts) +
      '<h2 style="margin-top:16px">Uncertainties</h2>' + renderList(data.uncertainties) + "</div>";
  }

  html += '<div class="panel"><h2>Provider status</h2>' + renderProviderStatus(data.provider_status) + "</div>";

  html += '<div class="panel"><h2>Run metadata</h2><div class="stat-grid">' +
    stat(m.search_iterations, "Iterations") +
    stat(m.sources_considered, "Considered") +
    stat(m.sources_selected, "Selected") +
    stat(m.queries_executed, "Queries") +
    stat(m.duplicates_merged, "Duplicates merged") +
    stat(m.citation_validation, "Citations") +
    stat(m.stop_reason, "Stop reason") +
    stat(m.elapsed_seconds + "s", "Elapsed") +
    "</div></div>";

  if (data.debug) {
    html += '<div class="panel"><details><summary>Raw debug info</summary>' +
      '<pre class="debug">' + escapeHtml(JSON.stringify(data.debug, null, 2)) + "</pre></details></div>";
  }

  document.getElementById("results").innerHTML = html;
}

async function runResearch() {
  var question = document.getElementById("question").value.trim();
  if (!question) {
    document.getElementById("statusText").textContent = "Enter a question first.";
    return;
  }

  var btn = document.getElementById("submitBtn");
  var spinner = document.getElementById("spinner");
  var statusText = document.getElementById("statusText");
  btn.disabled = true;
  spinner.classList.add("on");
  statusText.textContent = "Researching — this can take a while for multi-iteration runs...";
  document.getElementById("results").innerHTML = "";

  var body = {
    question: question,
    options: {
      max_iterations: parseInt(document.getElementById("maxIterations").value, 10),
      max_sources: parseInt(document.getElementById("maxSources").value, 10),
      include_debug: document.getElementById("includeDebug").checked
    }
  };

  try {
    var res = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    var data = await res.json();
    if (!res.ok) {
      document.getElementById("results").innerHTML =
        '<div class="panel"><div class="error-box">Request failed (' + res.status + "): " +
        escapeHtml(data.detail || JSON.stringify(data)) + "</div></div>";
    } else {
      renderResults(data);
    }
    statusText.textContent = "Done.";
  } catch (err) {
    document.getElementById("results").innerHTML =
      '<div class="panel"><div class="error-box">Network error: ' + escapeHtml(String(err)) + "</div></div>";
    statusText.textContent = "Failed.";
  } finally {
    btn.disabled = false;
    spinner.classList.remove("on");
  }
}

document.getElementById("question").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runResearch();
});
</script>
</body>
</html>
"""