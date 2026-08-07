from pathlib import Path
import argparse

from benchmark.evaluation.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    runner = BenchmarkRunner(Path(args.config))
    df = runner.run()

    print()
    print(df)


if __name__ == "__main__":
    main()