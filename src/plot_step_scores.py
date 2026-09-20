from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_step_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Algorithm:\s*(?P<algorithm>\w+)\s*,\s*Episode:\s*(?P<episode>\d+)\s*,\s*.*?Total Score:\s*(?P<score>[-+]?\d+)",
        re.IGNORECASE,
    )
    points = []
    for match in pattern.finditer(text):
        algorithm = match.group("algorithm").strip().lower()
        episode = int(match.group("episode"))
        score = int(match.group("score"))
        points.append((episode, score, algorithm))
    return points


def plot_group(files, title, output_name):
    series = {}
    for file_path in files:
        entries = parse_step_file(file_path)
        if not entries:
            continue

        for episode, score, algorithm in entries:
            if algorithm == "q_learning":
                label = "Q-Learning"
            elif algorithm == "sarsa":
                label = "SARSA"
            else:
                continue

            series.setdefault(label, {"episodes": [], "scores": []})
            series[label]["episodes"].append(episode)
            series[label]["scores"].append(score)

    if not series:
        raise ValueError(f"No valid Q-learning or SARSA data found for {output_name}")

    plt.figure(figsize=(14, 8))
    for label, values in series.items():
        plt.plot(
            values["episodes"],
            values["scores"],
            label=label,
            alpha=0.9,
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "results" / output_name
    plt.savefig(output_path, dpi=300)
    print(f"Saved chart to: {output_path}")
    plt.close()


def plot_average_group(files, title, output_name):
    episode_scores = {"Q-Learning": {}, "SARSA": {}}

    for file_path in files:
        entries = parse_step_file(file_path)
        if not entries:
            continue

        for episode, score, algorithm in entries:
            if algorithm == "q_learning":
                label = "Q-Learning"
            elif algorithm == "sarsa":
                label = "SARSA"
            else:
                continue

            episode_scores[label].setdefault(episode, []).append(score)

    plt.figure(figsize=(10, 6))
    for label in ["Q-Learning", "SARSA"]:
        if not episode_scores[label]:
            continue

        episodes = sorted(episode_scores[label])
        averages = [sum(episode_scores[label][ep]) / len(episode_scores[label][ep]) for ep in episodes]
        plt.plot(
            episodes,
            averages,
            label=label,
            linewidth=2.5,
            alpha=0.95,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Average Total Score", fontsize=11)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "results" / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved chart to: {output_path}")
    plt.close()


def plot_report_style_average(files, output_name):
    episode_scores = {"Q-Learning": {}, "SARSA": {}}

    for file_path in files:
        entries = parse_step_file(file_path)
        if not entries:
            continue

        for episode, score, algorithm in entries:
            if algorithm == "q_learning":
                label = "Q-Learning"
            elif algorithm == "sarsa":
                label = "SARSA"
            else:
                continue

            episode_scores[label].setdefault(episode, []).append(score)

    plt.figure(figsize=(8, 5))
    for label, color in [("Q-Learning", "#1f77b4"), ("SARSA", "#ff7f0e")]:
        if not episode_scores[label]:
            continue

        episodes = sorted(episode_scores[label])
        averages = [sum(episode_scores[label][ep]) / len(episode_scores[label][ep]) for ep in episodes]
        plt.plot(episodes, averages, label=label, color=color, linewidth=2.5)

    plt.title("Average Total Score Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "results" / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved chart to: {output_path}")
    plt.close()


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    files = sorted(results_dir.glob("*step*.txt"))

    if not files:
        raise FileNotFoundError(f"No step files found in: {results_dir}")

    one_time_files = sorted(results_dir.glob("1times_*step*.txt"))
    two_time_files = sorted(results_dir.glob("2times_*step*.txt"))

    if one_time_files:
        plot_group(one_time_files, "1times - Q-Learning vs SARSA Episode Score Trend", "1times_ql_vs_sarsa_step_scores.png")
    if two_time_files:
        plot_group(two_time_files, "2times - Q-Learning vs SARSA Episode Score Trend", "2times_ql_vs_sarsa_step_scores.png")

    combined_files = one_time_files + two_time_files
    if combined_files:
        plot_average_group(combined_files, "Combined Average - Q-Learning vs SARSA Episode Score Trend", "combined_average_ql_vs_sarsa_step_scores.png")
        plot_report_style_average(combined_files, "report_combined_average_ql_vs_sarsa_step_scores.png")


if __name__ == "__main__":
    main()
