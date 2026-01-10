import os
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    if is_win_rate_plot:
        index_value = 0
        while index_value < len(y_value_list):
            y_value_list[index_value] = y_value_list[index_value] * 100.0
            index_value += 1
        moving_average_window = 5
        y_value_list = compute_moving_average(y_value_list, moving_average_window)

    plt.figure()
    plt.plot(episode_list, y_value_list)
    plt.xlabel("episode")

    if is_win_rate_plot:
        plt.ylabel(y_column_name)
        plt.ylim(55, 90)
    else:
        plt.ylabel(y_column_name)

    plt.title(y_column_name + " vs episode")
    plt.grid(True)
    plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    csv_file_path = "training_log_DMC_800000_episodes.csv"

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
            print("Skipped (no valid data):", column_name)


if __name__ == "__main__":
    main()
