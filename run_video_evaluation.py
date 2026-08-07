from pathlib import Path
import argparse

from video_evaluation.evaluation.video_evaluation import evaluate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Pfad zur YAML-Konfigurationsdatei.")
    args = parser.parse_args()

    evaluate_from_config(args.config)


if __name__ == "__main__":
    main()