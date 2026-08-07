from pathlib import Path
import yaml
import pandas as pd
from mpmath.calculus.calculus import defun

from benchmark.core.cape import load_cape_gt
from benchmark.core.smpl import load_smpl_j_regressor
from benchmark.core.mesh_export import MeshExporter
from benchmark.trackers.registry import get_tracker_loader
from benchmark.evaluation.evaluator import SequenceEvaluator

class BenchmarkRunner:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.cape_root = Path(self.config["cape_root"])
        self.gt_type = self.config.get("gt_type", "unclothed")
        self.smpl_root = Path(self.config["smpl_root"])

        smplx2smpl_path = self.config.get("smplx2smpl_path", None)
        if smplx2smpl_path is not None:
            smplx2smpl_path = Path(smplx2smpl_path)

        self.output_root = Path(self.config["output"]["dir"])
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.output_root / self.config["output"]["summary_file"]

        smpl_male_path = self.smpl_root / "SMPL_MALE.pkl"
        smpl_female_path = self.smpl_root / "SMPL_FEMALE.pkl"

        j_regressor_male = load_smpl_j_regressor(smpl_male_path)
        j_regressor_female = load_smpl_j_regressor(smpl_female_path)
        self.evaluator = SequenceEvaluator(j_regressor_male, j_regressor_female)

        self.exporter = MeshExporter(smpl_male_path, smpl_female_path, self.output_root)
        self.export_meshes_all_sequences = self.config["output"].get("export_meshes_all_sequences", False)

        tracker_name = self.config["tracker"]["name"]
        self.mode = self.config["tracker"].get("mode", "default")
        self.tracker_loader = get_tracker_loader(tracker_name, smplx2smpl_path)

        self.metrics = self.config["metrics"]

    def run(self):
        rows = []

        for item in self.config["sequences"]:
            subject = item["subject"]
            sequence = item["sequence"]
            gender = item["gender"]
            export_meshes = item.get("export_meshes", False)

            if self.mode == "auto":
                prediction_paths = [Path(path) for path in item["prediction_paths"]]
            else:
                prediction_paths = [Path(item["prediction_path"])]


            print(f"Evaluating {subject} / {sequence}")

            try:
                gt = load_cape_gt(cape_root=self.cape_root, subject=subject, sequence=sequence, gt_type=self.gt_type, gender=gender)

                pred = self.tracker_loader.load(prediction_paths=prediction_paths, subject=subject, sequence=sequence)

                result = self.evaluator.evaluate(gt, pred, gender, self.mode)

                result["status"] = "ok"
                result["error"] = ""

                if export_meshes or self.export_meshes_all_sequences:
                    self.exporter.export_meshes(gt, result)

            except Exception as e:
                result = {
                    "subject": subject,
                    "sequence": sequence,
                    "tracker": self.tracker_loader.tracker_name,
                    "status": "failed",
                    "error": str(e),
                }

            selected_keys = ["subject", "sequence", "tracker", "num_matched_frames", *self.metrics, "status", "error"]

            rows.append({
                key: result.get(key)
                for key in selected_keys
            })

        df = pd.DataFrame(rows)
        numeric_cols = [metric for metric in self.metrics if metric in df.columns and metric[:9] != "per_frame"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        avg_row = {col: "" for col in df.columns}
        avg_row["subject"] = "AVERAGE"

        sd_row = {col: "" for col in df.columns}
        sd_row["subject"] = "STD"

        for col in numeric_cols:
            avg_row[col] = df[col].mean()
            sd_row[col] = df[col].std()

        df = pd.concat([df, pd.DataFrame([avg_row, sd_row])],ignore_index=True)


        df.to_csv(self.summary_file, index=False)

        return df
