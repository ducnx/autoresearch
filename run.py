"""
Multi-Agent Autoresearch — Main Entry Point

Runs the autonomous multi-agent research loop inspired by PaperOrchestra.
Orchestrates specialized agents (Director, Hypothesis, Literature,
Experiment, Analysis, Report) to iteratively improve a language model.

Usage:
    uv run run.py                          # auto-detect GPU, use defaults
    uv run run.py --tag apr11              # set experiment tag
    uv run run.py --dry-run                # no GPU needed (simulated results)
    uv run run.py --max-experiments 10     # stop after 10 experiments
    uv run run.py --cloud-model gemini/gemini-2.5-flash  # override cloud LLM
    uv run run.py --local-model qwen3:8b   # override local Ollama model
"""

import argparse
import asyncio
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from core.config import Config, LLMConfig
from core.workspace import Workspace, Hypothesis, ExperimentResult
from agents.director import DirectorAgent
from agents.hypothesis import HypothesisAgent
from agents.literature import LiteratureAgent
from agents.experiment import ExperimentAgent
from agents.analysis import AnalysisAgent
from agents.report import ReportAgent


# ─── Console formatting ─────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

def banner():
    print(f"""
{BOLD}{MAGENTA}╔══════════════════════════════════════════════════════════════╗
║                   🔬 AUTORESEARCH                            ║
║            Multi-Agent Autonomous Research Framework          ║
║         Inspired by PaperOrchestra (arXiv:2604.05018)        ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def phase_header(phase: str, emoji: str = "🔄"):
    print(f"\n{BOLD}{BLUE}{'─' * 60}")
    print(f" {emoji}  {phase}")
    print(f"{'─' * 60}{RESET}")


def status_line(key: str, value, color: str = ""):
    print(f"  {DIM}{key}:{RESET} {color}{value}{RESET}")


# ─── Setup ───────────────────────────────────────────────────────

def setup_branch(config: Config):
    """Create or checkout the experiment branch."""
    root = str(config.project_root)
    branch = f"autoresearch/{config.run_tag}"

    # Check if branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch],
        cwd=root, capture_output=True, text=True,
    )
    if branch in result.stdout:
        print(f"  Checking out existing branch: {branch}")
        subprocess.run(
            ["git", "checkout", branch],
            cwd=root, capture_output=True, check=True,
        )
    else:
        print(f"  Creating new branch: {branch}")
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=root, capture_output=True, check=True,
        )

    return branch


def verify_data(config: Config):
    """Check if training data and tokenizer exist."""
    import os
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    data_ok = os.path.exists(os.path.join(cache_dir, "data"))
    tok_ok = os.path.exists(os.path.join(cache_dir, "tokenizer"))

    if not data_ok or not tok_ok:
        print(f"  {YELLOW}⚠ Data not found at {cache_dir}")
        print(f"  Run 'uv run prepare.py' first to download data and train tokenizer{RESET}")
        if not config.dry_run:
            sys.exit(1)
        else:
            print(f"  {DIM}(Continuing in dry-run mode){RESET}")

    return data_ok and tok_ok


# ─── Main Research Loop ─────────────────────────────────────────

def run_research_loop(config: Config):
    """
    Main multi-agent research loop.

    Flow (inspired by PaperOrchestra's parallel + iterative pipeline):
    1. Director generates research brief
    2. Hypothesis + Literature agents run in parallel
    3. Director selects best hypothesis
    4. Experiment agent implements and runs
    5. Analysis agent interprets results
    6. Director decides keep/discard
    7. Report agent generates periodic reports
    8. Loop back to step 1
    """

    # Initialize workspace
    workspace = Workspace(config.workspace_dir)

    # Initialize agents
    director = DirectorAgent(config, workspace)
    hypothesis_agent = HypothesisAgent(config, workspace)
    literature_agent = LiteratureAgent(config, workspace)
    experiment_agent = ExperimentAgent(config, workspace)
    analysis_agent = AnalysisAgent(config, workspace)
    report_agent = ReportAgent(config, workspace)

    # Setup
    phase_header("Setup", "⚙️")
    branch = setup_branch(config)
    workspace.update_state(run_tag=config.run_tag, branch=branch)
    data_ok = verify_data(config)

    status_line("Branch", branch)
    status_line("GPU", "available" if config.has_gpu else "not detected")
    status_line("Dry run", str(config.dry_run))
    status_line("Ollama", "available" if config.has_ollama else "not detected")
    status_line("Project", config.project)
    status_line("Workspace", str(config.workspace_dir))

    # Print LLM configs
    print(f"\n  {BOLD}LLM Configuration:{RESET}")
    for agent_name in ["director", "hypothesis", "literature", "experiment", "analysis", "report"]:
        llm_cfg = getattr(config.llm, agent_name)
        provider = "☁️  cloud" if "ollama" not in llm_cfg.model else "🏠 local"
        status_line(f"  {agent_name}", f"{provider} → {llm_cfg.model}")

    # Read initial train.py
    train_path = config.project_dir / config.train_script
    train_code = train_path.read_text()

    # ─── Baseline run ────────────────────────────────────────────
    experiment_id = 0
    state = workspace.get_state()

    if state.get("baseline_bpb") is None:
        phase_header("Baseline Run", "📊")
        print("  Running baseline experiment (no modifications)...")

        baseline_hyp = Hypothesis(
            id="baseline",
            description="Baseline — unmodified train.py",
            predicted_impact="n/a",
            complexity="n/a",
            risk="low",
            category="baseline",
            rationale="Establish baseline performance",
        )

        result = experiment_agent.run(baseline_hyp, experiment_id)
        result.status = "keep"
        workspace.add_result(result)

        if result.val_bpb > 0:
            print(f"  {GREEN}✓ Baseline val_bpb: {result.val_bpb:.6f}{RESET}")
        else:
            print(f"  {RED}✗ Baseline run failed: {result.error_message}{RESET}")
            if not config.dry_run:
                print("  Fix the issue and re-run.")
                return

        experiment_id = 1

    else:
        experiment_id = state["experiment_count"]
        print(f"  Resuming from experiment {experiment_id} (baseline: {state['baseline_bpb']:.6f})")

    # ─── Main Loop ───────────────────────────────────────────────
    consecutive_failures = 0
    max_consecutive_failures = 5

    while True:
        # Check experiment limit
        if config.max_experiments and experiment_id >= config.max_experiments:
            print(f"\n{GREEN}Reached experiment limit ({config.max_experiments}). Stopping.{RESET}")
            break

        phase_header(f"Experiment {experiment_id}", "🧪")
        start_time = time.time()

        try:
            # ── Phase 1: Research Brief ──────────────────────────
            print(f"  {CYAN}[1/5] Director: generating research brief...{RESET}")
            research_brief = director.get_research_brief()

            # ── Phase 2: Parallel — Hypothesis + Literature ──────
            print(f"  {CYAN}[2/5] Parallel: generating hypotheses + searching literature...{RESET}")

            # Read current train.py (may have changed from previous experiments)
            train_code = train_path.read_text()

            # Run hypothesis and literature agents in parallel using threads
            with ThreadPoolExecutor(max_workers=2) as executor:
                hyp_future = executor.submit(
                    hypothesis_agent.run,
                    research_brief=research_brief,
                    train_code=train_code,
                )
                lit_future = executor.submit(
                    literature_agent.run,
                    research_brief=research_brief,
                )

                hypotheses = hyp_future.result()
                literature = lit_future.result()

            print(f"    Generated {len(hypotheses)} hypotheses, found {len(literature)} techniques")

            # ── Phase 3: Director Selects ────────────────────────
            print(f"  {CYAN}[3/5] Director: selecting best hypothesis...{RESET}")
            lit_dicts = [e.to_dict() for e in literature] if literature else None
            selected = director.select_hypothesis(hypotheses, lit_dicts)

            if not selected:
                print(f"  {YELLOW}⚠ No hypothesis selected, skipping...{RESET}")
                experiment_id += 1
                continue

            print(f"    Selected: {selected.description}")
            print(f"    [{selected.predicted_impact} impact, {selected.complexity} complexity, {selected.risk} risk]")

            # Update hypothesis status
            workspace.update_hypothesis_status(selected.id, "selected")

            # ── Phase 4: Experiment ──────────────────────────────
            print(f"  {CYAN}[4/5] Experiment: implementing and running...{RESET}")
            result = experiment_agent.run(selected, experiment_id)

            if result.status == "crash":
                print(f"  {RED}✗ Experiment crashed: {result.error_message[:100]}{RESET}")
                result.status = "crash"
                workspace.add_result(result)
                workspace.update_hypothesis_status(selected.id, "tested")

                # Revert changes
                experiment_agent.revert_experiment()
                consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n{RED}Too many consecutive failures ({max_consecutive_failures}). Stopping.{RESET}")
                    break

                experiment_id += 1
                continue

            consecutive_failures = 0

            print(f"    val_bpb: {result.val_bpb:.6f}")
            print(f"    VRAM: {result.peak_vram_mb / 1024:.1f} GB")
            print(f"    MFU: {result.mfu_percent:.1f}%")

            # ── Phase 5: Analysis + Decision ─────────────────────
            print(f"  {CYAN}[5/5] Analysis + Decision...{RESET}")
            analysis = analysis_agent.analyze(result)

            # Director makes final decision
            decision = director.decide(result, selected)
            result.status = decision
            workspace.add_result(result)
            workspace.update_hypothesis_status(selected.id, "tested")

            state = workspace.get_state()
            elapsed = time.time() - start_time

            if decision == "keep":
                improvement = ""
                if state.get("baseline_bpb"):
                    pct = (state["baseline_bpb"] - result.val_bpb) / state["baseline_bpb"] * 100
                    improvement = f" ({pct:.2f}% from baseline)"
                print(f"  {GREEN}✓ KEEP — val_bpb: {result.val_bpb:.6f}{improvement}{RESET}")
            else:
                print(f"  {RED}✗ DISCARD — val_bpb: {result.val_bpb:.6f}{RESET}")
                experiment_agent.revert_experiment()

            print(f"  {DIM}Elapsed: {elapsed:.0f}s{RESET}")

            # ── Periodic Report ──────────────────────────────────
            if experiment_id > 0 and experiment_id % config.report_interval == 0:
                phase_header("Progress Report", "📋")
                report_agent.generate_report()

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted by user. Generating final report...{RESET}")
            report_agent.generate_report()
            break
        except Exception as e:
            print(f"  {RED}Error in experiment loop: {e}{RESET}")
            import traceback
            traceback.print_exc()
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break

        experiment_id += 1

    # ─── Final Report ────────────────────────────────────────────
    phase_header("Final Report", "📊")
    report_agent.generate_report()

    state = workspace.get_state()
    print(f"\n{BOLD}Research Complete{RESET}")
    status_line("Total experiments", state["experiment_count"])
    status_line("Baseline val_bpb", state.get("baseline_bpb", "N/A"))
    status_line("Best val_bpb", state.get("best_bpb", "N/A"), GREEN)
    if state.get("baseline_bpb") and state.get("best_bpb"):
        improvement = (state["baseline_bpb"] - state["best_bpb"]) / state["baseline_bpb"] * 100
        status_line("Improvement", f"{improvement:.2f}%", GREEN)
    print(f"\n  Reports saved to: {config.workspace_dir / 'reports'}")


# ─── CLI ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Autoresearch — Autonomous AI Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run run.py                              # auto-detect everything
  uv run run.py --tag apr11                   # set experiment run tag
  uv run run.py --dry-run --max-experiments 5 # test without GPU
  uv run run.py --cloud-model gpt-4o          # use OpenAI for cloud agents
  uv run run.py --local-model llama3.2:3b     # use smaller Ollama model
        """,
    )

    parser.add_argument("--project", type=str, default="default",
                        help="Project name under projects/ directory")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment run tag (default: auto-generated from date)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate experiments without GPU")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Maximum number of experiments to run")
    parser.add_argument("--report-interval", type=int, default=5,
                        help="Generate report every N experiments")
    parser.add_argument("--cloud-model", type=str, default=None,
                        help="Cloud LLM model for Director + Literature agents")
    parser.add_argument("--local-model", type=str, default=None,
                        help="Local Ollama model for Hypothesis + Experiment + Analysis + Report agents")

    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    # Build config
    tag = args.tag or datetime.now().strftime("%b%d").lower()

    config = Config.from_env(
        project=args.project,
        run_tag=tag,
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
        report_interval=args.report_interval,
    )

    # Apply CLI overrides
    if args.cloud_model:
        config.llm.director = LLMConfig.cloud(model=args.cloud_model)
        config.llm.literature = LLMConfig.cloud(model=args.cloud_model)

    if args.local_model:
        if config.has_ollama:
            config.llm.hypothesis = LLMConfig.local(model=f"ollama/{args.local_model}")
            config.llm.experiment = LLMConfig.local(model=f"ollama/{args.local_model}")
            config.llm.analysis = LLMConfig.local(model=f"ollama/{args.local_model}")
            config.llm.report = LLMConfig.local(model=f"ollama/{args.local_model}")
        else:
            print(f"{YELLOW}⚠ Ollama not available, ignoring --local-model{RESET}")

    # Run
    run_research_loop(config)


if __name__ == "__main__":
    main()
