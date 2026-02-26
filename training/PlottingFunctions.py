import os
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def _gaussian_kernel(sigma, radius=6):
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _load_snap_cube_samples_csv(csv_path):
    df = pd.read_csv(csv_path)

    lower_cols = [str(c).strip().lower() for c in df.columns]
    if ("checkpoint" not in lower_cols) and (len(df.columns) >= 6):
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["checkpoint", "episode", "sample_idx", "cubes_abs", "winner", "cubes_net"],
        )

    return df


def _find_column(df, candidate_names):
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for name in candidate_names:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def _ensure_cubes_net_column(df):
    cubes_net_col = _find_column(df, ["cubes_net", "net_cubes", "cube_net", "net_cube"])
    if cubes_net_col is not None:
        return pd.to_numeric(df[cubes_net_col], errors="coerce")

    winner_col = _find_column(df, ["winner", "winner_name", "match_winner"])
    cubes_abs_col = _find_column(df, ["cubes_abs", "cubes", "cube_amount", "cube_value"])

    if winner_col is not None and cubes_abs_col is not None:
        cubes_abs = pd.to_numeric(df[cubes_abs_col], errors="coerce")
        winner = df[winner_col].astype(str).str.strip().str.lower()
        sign = np.where(winner.isin(["enemy", "opponent"]), -1.0, 1.0)
        return cubes_abs * sign

    raise ValueError(
        "Impossibile trovare 'cubes_net' e non riesco a ricostruirlo da 'winner' + 'cubes_abs'."
    )


def plot_cube_distributions_snap(
    csv_path,
    out_png_path,
    sigma=1.0,
    checkpoints_order=("untrained", "start", "oneThird", "twoThird", "end"),
):
    df = _load_snap_cube_samples_csv(csv_path)

    checkpoint_col = _find_column(df, ["checkpoint", "stage", "label"])
    if checkpoint_col is None:
        raise ValueError("Colonna checkpoint non trovata.")

    cubes_net = _ensure_cubes_net_column(df)

    df = df.copy()
    df["_cubes_net_plot"] = cubes_net
    df["_checkpoint_plot"] = df[checkpoint_col].astype(str).str.strip()

    df = df.dropna(subset=["_cubes_net_plot"])
    if df.empty:
        raise ValueError("Nessun dato valido per la distribuzione dei cubi.")

    bins = np.arange(-8.5, 8.6, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2.0
    kernel = _gaussian_kernel(sigma=sigma, radius=6)

    plt.figure(figsize=(10, 6))

    present_checkpoints = list(pd.unique(df["_checkpoint_plot"]))
    ordered = [c for c in checkpoints_order if c in present_checkpoints]
    ordered += [c for c in present_checkpoints if c not in ordered]

    for checkpoint in ordered:
        x = df.loc[df["_checkpoint_plot"] == checkpoint, "_cubes_net_plot"].to_numpy(dtype=float)
        if x.size == 0:
            continue

        hist, _ = np.histogram(x, bins=bins, density=False)
        hist = hist.astype(float) / float(hist.sum())

        smooth = np.convolve(hist, kernel, mode="same")
        plt.plot(centers, smooth, label=f"{checkpoint}")

    plt.axvline(0.0, linestyle="--", alpha=0.5)
    plt.xlabel("Cubi ottenuti per partita(valori negativi = cubi persi, valori positivi = cubi vinti)")
    plt.ylabel("Probabilità (appiattita)")
    plt.title("Distribuzione di cubi vinti o persi calcolata su più valutazioni")
    plt.xticks(np.arange(-8, 9, 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()


def compute_moving_average(value_list, window_size):
    smoothed_value_list = []
    index_value = 0
    while index_value < len(value_list):
        start_index = index_value - window_size + 1
        if start_index < 0:
            start_index = 0

        sum_value = 0.0
        count_value = 0
        inner_index = start_index
        while inner_index <= index_value:
            sum_value += float(value_list[inner_index])
            count_value += 1
            inner_index += 1

        smoothed_value_list.append(sum_value / float(count_value))
        index_value += 1

    return smoothed_value_list


def plot_training_csv_metric(csv_file_path, y_column_name, output_image_path):
    episode_list = []
    y_value_list = []

    with open(csv_file_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            row = dict(row)

            episode_text = row.get("episode", "")
            y_text = row.get(y_column_name, "")

            if episode_text is None or y_text is None:
                continue

            episode_text = str(episode_text).strip()
            y_text = str(y_text).strip()

            if episode_text == "" or y_text == "":
                continue

            try:
                episode_value = int(float(episode_text))
                y_value = float(y_text)
            except ValueError:
                continue

            episode_list.append(episode_value)
            y_value_list.append(y_value)

    if len(episode_list) == 0:
        raise ValueError("No valid numeric data to plot.")

    lower_csv_name = str(csv_file_path).lower()
    lower_column_name = str(y_column_name).lower()

    is_win_rate_plot = ("win rate" in lower_csv_name) or ("win_rate" in lower_column_name) or (
        "win rate" in lower_column_name)

    is_average_cubes_plot = (
        lower_column_name == "average_cubes_won"
        or "average_cubes_won" in lower_column_name
    )

    if is_win_rate_plot:
        index_value = 0
        while index_value < len(y_value_list):
            y_value_list[index_value] = y_value_list[index_value] * 100.0
            index_value += 1
        moving_average_window = 9
        y_value_list = compute_moving_average(y_value_list, moving_average_window)
    elif is_average_cubes_plot:
        moving_average_window = 9
        y_value_list = compute_moving_average(y_value_list, moving_average_window)

    plt.figure()
    plt.plot(episode_list, y_value_list)
    plt.xlabel("episode")

    if is_win_rate_plot:
        plt.ylabel(y_column_name)
        plt.ylim(40, 60)
    elif is_average_cubes_plot:
        plt.ylabel(y_column_name)
        plt.ylim(0.2, 0.9)
    else:
        plt.ylabel(y_column_name)

    plt.title(y_column_name + " vs episode")
    plt.grid(True)
    plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    plots_directory = "plots"
    os.makedirs(plots_directory, exist_ok=True)

    plot_cube_distributions_snap(
        csv_path="snap_agent_cubes_per_match.csv",
        out_png_path=os.path.join(plots_directory, "snap_agent_cubes_distribution.png"),
        sigma=1.0,
        checkpoints_order=("start", "mid", "end"),
    )

    print("Saved:", os.path.join(plots_directory, "snap_agent_cubes_distribution.png"))


'''
csv_file_path = "training_log_snap_DMC_1500005_episodes.csv"

plots_directory = "plots"
os.makedirs(plots_directory, exist_ok=True)

with open(csv_file_path, "r", newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    column_name_list = list(csv_reader.fieldnames or [])

if len(column_name_list) == 0:
    print("CSV vuoto o senza header.")
    return

base_name = os.path.splitext(os.path.basename(csv_file_path))[0]

excluded_columns = {"loss", "episode", "tie_rate", "enemy_win_rate", "final_reward_ally", "final_reward_enemy"}

for column_name in column_name_list:
    if column_name in excluded_columns:
        continue

    safe_column_name = column_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    output_image_path = os.path.join(plots_directory, base_name + "__" + safe_column_name + ".png")

    try:
        plot_training_csv_metric(csv_file_path, column_name, output_image_path)
        print("Saved:", output_image_path)
    except ValueError:
        print("Skipped (no valid data):", column_name)'''

if __name__ == "__main__":
    main()
