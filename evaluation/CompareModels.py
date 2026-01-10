import os
import torch

from training.TrainingNetwork import load_q_network
from training.TrainingUtilityFunctions import format_csv_value
from training.TrainingUtilityFunctions import get_input_dimension

from training.TrainingDeepMonteCarlo import evaluate_against_chosen_opponent

if __name__ == "__main__":
    episodes_list = [25000]

    number_of_games = 10000
    epsilon_agent = 0.0
    verbose = False

    input_dimension = get_input_dimension()
    hidden_dimension = 512

    print("=== Final evaluation vs random opponent (multi-model) ===")
    print(f"Games per model: {number_of_games}\n")

    for number_of_episodes in episodes_list:
        model_path = f"trained_q_network_{number_of_episodes}_episodes.pt"

        if not os.path.exists(model_path):
            print(f"[SKIP] Modello non trovato: {model_path}")
            continue

        loaded_q_network = load_q_network(
            model_path,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
        )

        with torch.no_grad():
            final_eval_results = evaluate_against_chosen_opponent(
                loaded_q_network,
                number_of_games=number_of_games,
                epsilon_agent=epsilon_agent,
                verbose=verbose,
            )

        print(f"\n--- Model trained for {number_of_episodes} episodes ---")
        for key, value in final_eval_results.items():
            print(f"{key}: {format_csv_value(value)}")
