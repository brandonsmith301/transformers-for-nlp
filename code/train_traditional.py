import os
import time
import json
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
from datetime import datetime

import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_model_instance(model_name):
    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=330, max_iter=1000)
    elif model_name == "MultinomialNB":
        return MultinomialNB()
    elif model_name == "LinearSVC":
        return LinearSVC(random_state=330, dual=True, max_iter=1000)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=330)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


@hydra.main(
    version_base=None,
    config_path="../conf/traditional/",
    config_name="config_traditional",
)
def run_training(config):
    print_line = lambda: print("#" + "~~" * 50 + "#")
    start_time = time.time()

    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print_line()

    # --- Load Data ---
    print("Loading IMDB dataset...")
    dataset = datasets.load_dataset("imdb")
    dftr = dataset["train"].to_pandas()
    dfte = dataset["test"].to_pandas()
    print_line()

    # --- Prepare Data & Features ---
    print_line()

    vectorizer = TfidfVectorizer()  # - use default parameters

    print("Fitting TF-IDF on training data...")
    x_train = vectorizer.fit_transform(dftr["text"])
    print("Transforming test data...")
    x_test = vectorizer.transform(dfte["text"])
    print(f"TF-IDF feature shape: {x_train.shape}")
    print_line()

    # --- Train and Evaluate Models ---
    results = {}
    os.makedirs(config.outputs.results_dir, exist_ok=True)

    for model_name in config.models:
        model_start_time = time.time()
        print(f"--- Running Model: {model_name} ---")

        model = get_model_instance(model_name)

        print("Training model...")
        model.fit(x_train, dftr["label"])
        train_time = time.time() - model_start_time
        print(f"Training completed in {train_time:.2f}s")

        print("Evaluating model...")
        eval_start_time = time.time()
        y_pred = model.predict(x_test)
        eval_time = time.time() - eval_start_time

        accuracy = accuracy_score(dfte["label"], y_pred)

        print(f"Evaluation completed in {eval_time:.2f}s")
        print(f"Accuracy: {accuracy:.4f}")

        results[model_name] = {
            "accuracy": accuracy,
            "training_time_seconds": train_time,
            "evaluation_time_seconds": eval_time,
            "model_params": model.get_params(),
        }
        print_line()

    # --- Save Results ---
    total_time = time.time() - start_time
    results["overall_benchmark_time_seconds"] = total_time
    results["timestamp"] = datetime.now().isoformat()

    results_path = os.path.join(config.outputs.results_dir, config.outputs.log_file)
    print(f"Saving benchmark results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark finished in {total_time:.2f}s")


if __name__ == "__main__":
    run_training()
