"""
Autoresearch Dashboard — Real-time experiment monitoring.

A simple web dashboard for monitoring the multi-agent research loop.
Serves a single-page application that auto-refreshes with experiment
results, agent activity, and progress charts.

Usage:
    uv run dashboard.py                    # default port 8501
    uv run dashboard.py --port 8080        # custom port
"""

import argparse
import json
import http.server
import os
import urllib.parse
from datetime import datetime
from pathlib import Path


PROJECTS_DIR = Path(__file__).parent / "projects"
WORKSPACE_DIR = PROJECTS_DIR



def get_state(project="default"):
    WORKSPACE_DIR = PROJECTS_DIR / project / "workspace"
    """Read workspace state."""
    state_path = WORKSPACE_DIR / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def get_results(project="default"):
    WORKSPACE_DIR = PROJECTS_DIR / project / "workspace"
    """Read experiment results."""
    results_path = WORKSPACE_DIR / "results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())
    return []


def get_agent_logs(project="default"):
    WORKSPACE_DIR = PROJECTS_DIR / project / "workspace"
    """Read recent agent logs."""
    logs_dir = WORKSPACE_DIR / "logs"
    all_logs = []
    if logs_dir.exists():
        for log_file in sorted(logs_dir.glob("*.jsonl")):
            with open(log_file) as f:
                for line in f.readlines()[-20:]:
                    try:
                        all_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return all_logs[:50]


def build_html(project="default"):
    """Build the dashboard HTML."""
    state = get_state(project)
    results = get_results(project)
    logs = get_agent_logs(project)


    # Get available projects
    available_projects = []
    if  PROJECTS_DIR.exists():
        available_projects = sorted([d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()])
    else:
        available_projects = ["default"]
        
    project_options = ""
    for p in available_projects:
        selected = "selected" if p == project else ""
        project_options += f'<option value="{p}" {selected}>{p}</option>'

    baseline_bpb = state.get("baseline_bpb", "—")
    best_bpb = state.get("best_bpb", "—")
    experiment_count = state.get("experiment_count", 0)
    run_tag = state.get("run_tag", "—")

    # Calculate stats
    kept = sum(1 for r in results if r.get("status") == "keep")
    discarded = sum(1 for r in results if r.get("status") == "discard")
    crashed = sum(1 for r in results if r.get("status") == "crash")
    success_rate = f"{kept / max(experiment_count, 1) * 100:.0f}%" if experiment_count > 0 else "—"

    improvement = "—"
    if isinstance(baseline_bpb, (int, float)) and isinstance(best_bpb, (int, float)):
        improvement = f"{(baseline_bpb - best_bpb) / baseline_bpb * 100:.2f}%"

    # Build results rows
    results_rows = ""
    for r in reversed(results):
        status = r.get("status", "unknown")
        status_color = {"keep": "#22c55e", "discard": "#ef4444", "crash": "#f59e0b"}.get(status, "#888")
        status_badge = f'<span class="badge" style="background:{status_color}">{status}</span>'
        vram = f"{r.get('peak_vram_mb', 0) / 1024:.1f}" if r.get("peak_vram_mb", 0) > 0 else "—"
        results_rows += f"""
        <tr>
            <td>{r.get('experiment_id', '—')}</td>
            <td>{status_badge}</td>
            <td class="mono">{r.get('val_bpb', 0):.6f}</td>
            <td>{vram}</td>
            <td>{r.get('num_params_m', 0):.1f}M</td>
            <td class="desc">{r.get('description', '—')}</td>
        </tr>"""

    # Build chart data
    chart_labels = json.dumps([f"Exp {r.get('experiment_id', '?')}" for r in results if r.get("status") != "crash"])
    chart_values = json.dumps([r.get("val_bpb", 0) for r in results if r.get("status") != "crash"])
    chart_colors = json.dumps(["#22c55e" if r.get("status") == "keep" else "#ef4444" for r in results if r.get("status") != "crash"])

    # Build best-so-far line
    best_so_far = []
    current_best = float("inf")
    for r in results:
        if r.get("status") != "crash" and r.get("val_bpb", float("inf")) < current_best:
            current_best = r["val_bpb"]
        if r.get("status") != "crash":
            best_so_far.append(current_best)
    chart_best = json.dumps(best_so_far)

    # Build log entries
    log_html = ""
    agent_colors = {
        "director": "#6366f1", "hypothesis": "#f59e0b", "literature": "#10b981",
        "experiment": "#ef4444", "analysis": "#8b5cf6", "report": "#06b6d4",
    }
    for log in logs[:30]:
        color = agent_colors.get(log.get("agent", ""), "#888")
        ts = log.get("timestamp", "")[:19]
        log_html += f"""
        <div class="log-entry">
            <span class="log-time">{ts}</span>
            <span class="log-agent" style="color:{color}">[{log.get('agent', '?')}]</span>
            <span class="log-msg">{log.get('message', '')}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="15;url=/?project={project}">
    <title>Autoresearch Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --accent: #6366f1;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #f59e0b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 1.5rem;
            min-height: 100vh;
        }}
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }}
        .subtitle {{
            color: var(--text-dim);
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
        .stat-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
        }}
        .stat-card .label {{ font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-card .value {{ font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem; }}
        .stat-card .value.green {{ color: var(--green); }}
        .chart-container {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th {{
            text-align: left;
            padding: 0.6rem 0.75rem;
            color: var(--text-dim);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        td.desc {{ white-space: normal; max-width: 300px; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 600;
            color: #000;
        }}
        .panel {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        }}
        .panel h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-dim);
        }}
        .log-entry {{
            font-size: 0.78rem;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            font-family: 'JetBrains Mono', monospace;
        }}
        .log-time {{ color: var(--text-dim); margin-right: 0.5rem; }}
        .log-agent {{ font-weight: 600; margin-right: 0.5rem; }}
        .log-msg {{ color: var(--text); }}
        .footer {{ text-align: center; color: var(--text-dim); font-size: 0.75rem; margin-top: 2rem; }}
    </style>
</head>
<body>
    
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
        <h1>🔬 Autoresearch Dashboard</h1>
        <select onchange="window.location.href='/?project=' + this.value" style="padding: 0.5rem; border-radius: 8px; background: var(--card); color: var(--text); border: 1px solid var(--border);">
            {project_options}
        </select>
    </div>

    <p class="subtitle">Multi-Agent Autonomous Research · Run: <strong>{run_tag}</strong> · Auto-refreshes every 15s</p>

    <div class="grid">
        <div class="stat-card"><div class="label">Experiments</div><div class="value">{experiment_count}</div></div>
        <div class="stat-card"><div class="label">Baseline BPB</div><div class="value mono">{baseline_bpb if isinstance(baseline_bpb, str) else f'{baseline_bpb:.4f}'}</div></div>
        <div class="stat-card"><div class="label">Best BPB</div><div class="value green mono">{best_bpb if isinstance(best_bpb, str) else f'{best_bpb:.4f}'}</div></div>
        <div class="stat-card"><div class="label">Improvement</div><div class="value green">{improvement}</div></div>
        <div class="stat-card"><div class="label">Kept / Discarded / Crashed</div><div class="value">{kept} / {discarded} / {crashed}</div></div>
        <div class="stat-card"><div class="label">Success Rate</div><div class="value">{success_rate}</div></div>
    </div>

    <div class="chart-container">
        <canvas id="bpbChart" height="80"></canvas>
    </div>

    <div class="two-col">
        <div class="panel">
            <h3>📊 Experiment Results</h3>
            <div style="overflow-x:auto">
                <table>
                    <thead><tr><th>ID</th><th>Status</th><th>val_bpb</th><th>VRAM (GB)</th><th>Params</th><th>Description</th></tr></thead>
                    <tbody>{results_rows}</tbody>
                </table>
            </div>
        </div>
        <div class="panel">
            <h3>📝 Agent Activity</h3>
            <div style="max-height:400px;overflow-y:auto">{log_html}</div>
        </div>
    </div>

    <p class="footer">Autoresearch Multi-Agent Framework · Inspired by PaperOrchestra (arXiv:2604.05018)</p>

    <script>
        const ctx = document.getElementById('bpbChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_labels},
                datasets: [
                    {{
                        label: 'val_bpb',
                        data: {chart_values},
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        pointBackgroundColor: {chart_colors},
                        pointBorderColor: '#fff',
                        pointRadius: 5,
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true,
                    }},
                    {{
                        label: 'Best so far',
                        data: {chart_best},
                        borderColor: '#22c55e',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Validation BPB Over Experiments (Lower is Better)', color: '#e2e8f0' }},
                    legend: {{ labels: {{ color: '#94a3b8' }} }},
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, reverse: false }},
                }}
            }}
        }});
    </script>
</body>
</html>"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""


    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        project = qs.get("project", ["default"])[0]
        
        if parsed.path == "/" or parsed.path == "/index.html":
            html = build_html(project)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(get_state(project)).encode("utf-8"))
        elif parsed.path == "/api/results":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(get_results(project)).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Autoresearch Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
    args = parser.parse_args()

    server = http.server.HTTPServer(("", args.port), DashboardHandler)
    print(f"🔬 Autoresearch Dashboard running at http://localhost:{args.port}")
    print(f"   Workspace: {WORKSPACE_DIR}")
    print(f"   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
        server.shutdown()


if __name__ == "__main__":
    main()
