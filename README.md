# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## What's New: Multi-Agent Framework

This fork extends Karpathy's original autoresearch with a **multi-agent framework** inspired by [PaperOrchestra](https://arxiv.org/abs/2604.05018) (arXiv:2604.05018). Instead of a single AI agent running experiments in a loop, this version decomposes the research process into **6 specialized agents** that collaborate:

| Agent | Role | LLM |
|-------|------|-----|
| 🎯 **Research Director** | Orchestrates strategy, selects hypotheses, makes keep/discard decisions | Cloud API |
| 💡 **Hypothesis Agent** | Generates ranked experiment ideas with impact/risk assessment | Local (Ollama) |
| 📚 **Literature Agent** | Searches for relevant techniques and recent advances | Cloud API |
| 🔧 **Experiment Agent** | Implements code changes and runs training | Local (Ollama) |
| 📊 **Analysis Agent** | Interprets results, identifies patterns across experiments | Local (Ollama) |
| 📋 **Report Agent** | Generates progress reports with plots and summaries | Local (Ollama) |

### Key Design Principles (from PaperOrchestra)

- **Specialized agents** — each agent has a focused role with tailored prompts
- **Parallel execution** — Hypothesis + Literature agents run simultaneously
- **Iterative refinement** — Analysis feeds back into the next hypothesis cycle
- **Shared workspace** — agents communicate via structured JSON files
- **Hybrid LLM strategy** — cloud API for internet tasks, local Ollama for cost-sensitive tasks

## How it works

The repo has three layers:

1. **Experiment infrastructure** (`projects/<name>/project.json`, project entrypoints, datasets) — each project declares its editable files, run command, metric, and writing directory.
2. **Agent framework** (`agents/`, `core/`, `.agents/`) — the multi-agent system that drives autonomous research.
3. **Entry points** (`run.py`, `dashboard.py`) — CLI launcher and web dashboard.

### The Research Loop

```
┌─ Phase 1: Research Brief ──────────── Director generates strategy
│
├─ Phase 2: Parallel Research ─────────┐
│   ├─ Hypothesis Agent ──────────────── Generates 3-5 ranked ideas
│   └─ Literature Agent ──────────────── Searches for relevant techniques
│                                       └──────────────────────────────┘
├─ Phase 3: Selection ──────────────── Director picks best hypothesis
│
├─ Phase 4: Experiment ─────────────── Experiment Agent implements & runs
│
├─ Phase 5: Analysis + Decision ────── Analysis Agent interprets results
│                                      Director decides keep/discard
│
├─ Phase 6: Report (every 5 exps) ──── Report Agent generates progress report
│
└─ Loop ─────────────────────────────── Back to Phase 1
```

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), an LLM (cloud API key and/or [Ollama](https://ollama.ai)).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
# Only needed if you have a GPU and want to run real experiments
cd projects/default
uv run prepare.py
cd ../..

# 4. Set up your LLM
# Option A: Cloud API (set your API key)
export GEMINI_API_KEY="your-key-here"

# Option B: Local Ollama (install and pull a model)
# ollama pull qwen3:8b

# 5. Run the multi-agent framework
uv run run.py --tag my_first_run

# Or run in dry-run mode (no GPU needed):
uv run run.py --dry-run --max-experiments 5
```

## Running the dashboard

In a separate terminal:

```bash
uv run dashboard.py
# Open http://localhost:8501
```

## CLI options

```bash
uv run run.py                                    # auto-detect GPU, use defaults
uv run run.py --project my_project                # run for a specific project folder
uv run run.py --tag apr11                         # set experiment tag
uv run run.py --dry-run                           # simulate without GPU
uv run run.py --max-experiments 20                # stop after 20 experiments
uv run run.py --project wind-turbine              # run the CARE wind-turbine project
uv run run.py --cloud-model gpt-4o                # use OpenAI for cloud agents
uv run run.py --local-model llama3.2:3b           # use smaller Ollama model
uv run run.py --report-interval 3                 # report every 3 experiments
```

## Project contract

Each research project can include a `project.json` file. The framework uses it to decide:

- which files agents may edit,
- which files provide research context,
- which command runs an experiment,
- which metric is optimized and whether it is minimized or maximized,
- which data paths must exist,
- where results should be appended in LyX/LaTeX writing files.

The default project keeps the original `train.py` workflow. The `projects/wind-turbine` project exposes `energy_fault_detector` from its `src/` folder, runs `projects/wind-turbine/train.py`, reports `care_loss`, and appends experiment outcomes to `projects/wind-turbine/writing/result.lyx`.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | API key for Gemini (cloud agents) | — |
| `OPENAI_API_KEY` | API key for OpenAI (if using gpt-4o) | — |
| `AUTORESEARCH_CLOUD_MODEL` | Override cloud LLM model | `gemini/gemini-2.5-flash` |
| `AUTORESEARCH_LOCAL_MODEL` | Override local Ollama model | `qwen3:8b` |
| `AUTORESEARCH_API_KEY` | Override API key for cloud agents | — |

## Project structure

```
autoresearch/
├── agents/                     # Multi-agent framework
│   ├── base.py                 # Base agent class with LLM interface
│   ├── director.py             # Research Director (orchestrator)
│   ├── hypothesis.py           # Hypothesis generation
│   ├── literature.py           # Literature search
│   ├── experiment.py           # Code modification & training
│   ├── analysis.py             # Result interpretation
│   ├── report.py               # Progress reports & plots
│   └── writer.py               # LyX/LaTeX writing updates
├── core/                       # Infrastructure
│   ├── config.py               # Configuration & LLM settings
│   ├── project.py              # Per-project run/edit/write contract
│   ├── workspace.py            # Shared state management
│   └── runner.py               # Experiment runner (subprocess)
├── projects/                   # Research project directories
│   ├── default/                # Default project
│   │   ├── project.json        # Editable files, command, metric
│   │   ├── prepare.py          # Data prep & tokenizer
│   │   └── train.py            # Model & training loop
│   └── wind-turbine/           # CARE wind-turbine project
│       ├── project.json        # CARE command, metric, writing config
│       ├── train.py            # Project entrypoint
│       ├── src/                # energy_fault_detector package
│       └── writing/            # LyX paper files
├── .agents/                    # Agent system prompts (Markdown)
│   ├── director.md
│   ├── hypothesis.md
│   ├── literature.md
│   ├── experiment.md
│   ├── analysis.md
│   └── report.md
├── run.py                      # Main entry point
├── dashboard.py                # Web monitoring dashboard
├── program.md                  # Agent instructions (legacy)
└── pyproject.toml              # Dependencies
```

## Design choices

- **Hybrid LLM strategy.** Cloud API (Gemini/GPT-4o) for tasks requiring internet access or strong reasoning (Director, Literature), local Ollama for cost-sensitive frequent tasks (Hypothesis, Experiment, Analysis, Report). If a cloud call fails or runs out of quota, agents try a local Ollama fallback when available; if Ollama is unavailable, local agents route to cloud.
- **Flexible GPU support.** Auto-detects NVIDIA GPU. Falls back to `--dry-run` mode with simulated results for framework testing without GPU.
- **Shared workspace.** Agents communicate via JSON files in `workspace/`, not direct messages. This enables debugging, replay, and persistence across restarts.
- **Project-specific experiment infrastructure.** The original default `prepare.py` and `train.py` flow still works, while package-style projects can declare their own entrypoints and metrics.

## Attribution

- Original autoresearch by [Andrej Karpathy](https://github.com/karpathy/autoresearch)
- Multi-agent architecture inspired by [PaperOrchestra](https://arxiv.org/abs/2604.05018) (Song et al., 2026)

## License

MIT
