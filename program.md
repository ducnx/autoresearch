# autoresearch — multi-agent program

This is the program specification for the multi-agent autoresearch framework.

## Architecture

The framework uses 6 specialized AI agents that collaborate to run autonomous
research experiments, inspired by PaperOrchestra (arXiv:2604.05018).

### Agents

1. **Research Director** — Orchestrates the research loop. Generates research
   briefs, selects hypotheses, and makes keep/discard decisions. Uses cloud
   LLM for strong reasoning.

2. **Hypothesis Agent** — Generates ranked experiment ideas. Each hypothesis
   includes predicted impact, complexity, and risk. Uses local Ollama for
   cost efficiency (runs frequently).

3. **Literature Agent** — Searches for relevant techniques and recent advances
   in ML training. Uses cloud LLM for internet knowledge access. Runs in
   parallel with the Hypothesis Agent.

4. **Experiment Agent** — Translates selected hypotheses into code changes in
   `train.py`, manages git commits, and runs training experiments. Uses local
   Ollama for code generation.

5. **Analysis Agent** — Interprets experiment results in context of all prior
   experiments. Identifies patterns, detects diminishing returns, and suggests
   pivot strategies. Uses local Ollama.

6. **Report Agent** — Generates progress reports with matplotlib plots,
   markdown summaries, and experiment tables. Uses local Ollama.

## The Research Loop

```
LOOP FOREVER:
  1. Director generates research brief (what's working, what to try)
  2. PARALLEL:
     - Hypothesis Agent generates 3-5 ranked ideas
     - Literature Agent searches for relevant techniques
  3. Director selects best hypothesis (considering literature findings)
  4. Experiment Agent implements changes + runs training (~5 min)
  5. Analysis Agent interprets results
  6. Director decides: keep (advance branch) or discard (revert)
  7. Every 5 experiments: Report Agent generates progress report
  8. Go to 1
```

## Communication

Agents communicate via structured JSON files in `workspace/`:
- `workspace/state.json` — current research state
- `workspace/results.json` — all experiment results
- `workspace/hypotheses.json` — hypothesis queue
- `workspace/literature.json` — collected literature findings
- `workspace/logs/` — agent activity logs
- `workspace/reports/` — generated reports and plots

## Constraints

**What the agents CAN do:**
- Modify `train.py` — architecture, optimizer, hyperparameters, everything
- Run training experiments via `uv run train.py`
- Search for techniques and generate hypotheses
- Analyze results and generate reports

**What the agents CANNOT do:**
- Modify `prepare.py` — it's read-only
- Install new dependencies
- Change the evaluation metric
- Exceed the 5-minute training time budget

## Running

```bash
# Full autonomous mode (with GPU)
uv run run.py --tag apr11

# Dry-run mode (no GPU, simulated results)
uv run run.py --dry-run --max-experiments 10

# With custom LLM
uv run run.py --cloud-model gpt-4o --local-model llama3.2:3b
```

## Dashboard

```bash
uv run dashboard.py
# Open http://localhost:8501
```

The dashboard shows real-time experiment progress, val_bpb charts,
agent activity logs, and experiment results tables. Auto-refreshes
every 15 seconds.
