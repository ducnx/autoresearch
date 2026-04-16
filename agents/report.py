"""
Report Agent — generates progress reports and plots.

Analogous to PaperOrchestra's Plotting Agent.
Creates visual reports summarizing experiment progress
using matplotlib for plots and markdown for text.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.base import BaseAgent


class ReportAgent(BaseAgent):
    name = "report"
    role = "Report Agent — generates progress reports and visualizations"

    def generate_report(self) -> dict:
        """
        Generate a comprehensive progress report.

        Returns:
            Report dictionary with summary, findings, and plot data
        """
        self._log("Generating progress report...")

        results = self.workspace.get_results()
        state = self.workspace.get_state()

        if not results:
            self._log("No results to report on")
            return {"report": {"title": "No experiments completed yet"}}

        context = self.get_context()

        results_text = "\n".join([
            f"- Exp {r.experiment_id} [{r.status}]: val_bpb={r.val_bpb:.6f}, "
            f"VRAM={r.peak_vram_mb:.0f}MB, params={r.num_params_m:.1f}M — {r.description}"
            for r in results
        ])

        prompt = (
            f"## All Experiment Results\n{results_text}\n\n"
            f"## State\n"
            f"- Baseline: {state.get('baseline_bpb')}\n"
            f"- Best: {state.get('best_bpb')}\n"
            f"- Total experiments: {state.get('experiment_count')}\n\n"
            f"Generate a concise progress report. Include:\n"
            f"1. Executive summary (2-3 sentences)\n"
            f"2. Key findings (top 3-5)\n"
            f"3. Best result details\n"
            f"4. Success rate\n"
            f"5. Recommendations for next steps\n\n"
            f"Respond with JSON containing: report, plot_data"
        )

        messages = self._build_messages(prompt, context=context)

        try:
            report_data = self._call_llm_json(messages, temperature=0.3)
        except Exception as e:
            self._log(f"Error generating report: {e}")
            report_data = self._generate_fallback_report(results, state)

        # Generate plots
        try:
            self._generate_plots(results)
        except Exception as e:
            self._log(f"Error generating plots: {e}")

        # Save report as markdown
        try:
            self._save_report_markdown(report_data, results)
        except Exception as e:
            self._log(f"Error saving report: {e}")

        self._log("Report generated successfully")
        return report_data

    def _generate_fallback_report(self, results, state) -> dict:
        """Generate a basic report without LLM."""
        kept = [r for r in results if r.status == "keep"]
        discarded = [r for r in results if r.status == "discard"]
        crashed = [r for r in results if r.status == "crash"]

        best = min(kept, key=lambda r: r.val_bpb) if kept else None

        return {
            "report": {
                "title": f"Autoresearch Progress — {len(results)} Experiments",
                "summary": (
                    f"Completed {len(results)} experiments. "
                    f"{len(kept)} kept, {len(discarded)} discarded, {len(crashed)} crashed. "
                    f"Best val_bpb: {best.val_bpb:.6f}." if best else "No successful experiments."
                ),
                "key_findings": [f"Best result: {best.description}" if best else "No results yet"],
                "best_result": {
                    "val_bpb": best.val_bpb if best else None,
                    "experiment_id": best.experiment_id if best else None,
                    "description": best.description if best else None,
                } if best else None,
                "total_experiments": len(results),
                "success_rate": len(kept) / max(len(results), 1),
            },
            "plot_data": {
                "x_labels": [f"exp_{r.experiment_id}" for r in results if r.status != "crash"],
                "val_bpb_values": [r.val_bpb for r in results if r.status != "crash"],
                "statuses": [r.status for r in results if r.status != "crash"],
            },
        }

    def _generate_plots(self, results):
        """Generate matplotlib plots of experiment progress."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reports_dir = self.workspace.root / "reports"
        reports_dir.mkdir(exist_ok=True)

        # Filter out crashes
        valid = [r for r in results if r.status != "crash"]
        if not valid:
            return

        # --- Plot 1: val_bpb over experiments ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Autoresearch Progress", fontsize=14, fontweight="bold")

        # val_bpb trajectory
        ax = axes[0, 0]
        exp_ids = [r.experiment_id for r in valid]
        bpb_values = [r.val_bpb for r in valid]
        colors = ["#22c55e" if r.status == "keep" else "#ef4444" for r in valid]
        ax.scatter(exp_ids, bpb_values, c=colors, s=60, zorder=5, edgecolors="white", linewidth=0.5)
        ax.plot(exp_ids, bpb_values, "k-", alpha=0.3, linewidth=1)

        # Best line
        best_so_far = []
        current_best = float("inf")
        for r in valid:
            if r.val_bpb < current_best:
                current_best = r.val_bpb
            best_so_far.append(current_best)
        ax.plot(exp_ids, best_so_far, "b--", alpha=0.6, linewidth=1.5, label="Best so far")

        ax.set_xlabel("Experiment")
        ax.set_ylabel("val_bpb")
        ax.set_title("Validation BPB (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # VRAM usage
        ax = axes[0, 1]
        vram_gb = [r.peak_vram_mb / 1024 for r in valid]
        ax.bar(exp_ids, vram_gb, color=colors, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Peak VRAM (GB)")
        ax.set_title("Memory Usage")
        ax.grid(True, alpha=0.3, axis="y")

        # MFU
        ax = axes[1, 0]
        mfu_values = [r.mfu_percent for r in valid]
        ax.bar(exp_ids, mfu_values, color="#6366f1", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("MFU (%)")
        ax.set_title("Model FLOPs Utilization")
        ax.grid(True, alpha=0.3, axis="y")

        # Status distribution
        ax = axes[1, 1]
        statuses = {"keep": 0, "discard": 0, "crash": 0}
        for r in results:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        status_colors = {"keep": "#22c55e", "discard": "#ef4444", "crash": "#f59e0b"}
        labels = [k for k, v in statuses.items() if v > 0]
        values = [v for v in statuses.values() if v > 0]
        pie_colors = [status_colors[l] for l in labels]
        ax.pie(values, labels=labels, colors=pie_colors, autopct="%1.0f%%", startangle=90)
        ax.set_title("Experiment Outcomes")

        plt.tight_layout()
        plot_path = reports_dir / "progress.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        self._log(f"Plot saved to {plot_path}")

    def _save_report_markdown(self, report_data: dict, results):
        """Save report as a markdown file."""
        reports_dir = self.workspace.root / "reports"
        reports_dir.mkdir(exist_ok=True)

        report = report_data.get("report", {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_lines = [
            f"# {report.get('title', 'Autoresearch Report')}",
            f"",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"",
            f"## Summary",
            f"",
            report.get("summary", "No summary available."),
            f"",
        ]

        # Key findings
        findings = report.get("key_findings", [])
        if findings:
            md_lines.append("## Key Findings\n")
            for finding in findings:
                md_lines.append(f"- {finding}")
            md_lines.append("")

        # Results table
        md_lines.append("## Experiment Results\n")
        md_lines.append("| ID | Status | val_bpb | VRAM (GB) | Params (M) | Description |")
        md_lines.append("|-----|--------|---------|-----------|------------|-------------|")
        for r in results:
            vram = f"{r.peak_vram_mb / 1024:.1f}" if r.peak_vram_mb > 0 else "N/A"
            params = f"{r.num_params_m:.1f}" if r.num_params_m > 0 else "N/A"
            md_lines.append(
                f"| {r.experiment_id} | {r.status} | {r.val_bpb:.6f} | {vram} | {params} | {r.description} |"
            )
        md_lines.append("")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            md_lines.append("## Recommendations\n")
            for rec in recommendations:
                md_lines.append(f"- {rec}")
            md_lines.append("")

        # Progress plot
        plot_path = reports_dir / "progress.png"
        if plot_path.exists():
            md_lines.append("## Progress Chart\n")
            md_lines.append(f"![Progress](progress.png)\n")

        # Write markdown
        report_path = reports_dir / f"report_{timestamp}.md"
        report_path.write_text("\n".join(md_lines))

        # Also write a "latest" symlink
        latest_path = reports_dir / "latest_report.md"
        latest_path.write_text("\n".join(md_lines))

        self._log(f"Report saved to {report_path}")

    def run(self, **kwargs) -> dict:
        """Main entry point."""
        return self.generate_report()
