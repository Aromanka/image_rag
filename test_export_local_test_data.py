"""Tests for portable local-test dataset exports."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from utils.export_local_test_data import (
    PROJECT_ROOT,
    export_subsets,
    parse_args,
    select_samples,
)
from utils.local_test_data import (
    INSPECSAFE_SAFETY_LEVEL_DATASET,
    LABSAFETY_GEN_DATASET,
    DisplaySample,
    load_display_samples,
)


class ExportLocalTestDataTests(unittest.TestCase):
    def test_cli_defaults_to_combined_local_test_batch(self) -> None:
        with patch("sys.argv", ["export_local_test_data.py"]):
            args = parse_args()
        self.assertEqual(args.output_dir, PROJECT_ROOT / "data" / "local_test_batch")
        self.assertIsNone(args.dataset)

    def _image(self, path: Path, color: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), color).save(path)

    def _source_datasets(self, root: Path) -> tuple[Path, Path, Path, Path]:
        inspec_root = root / "inspec_images"
        inspec_annotations = root / "inspec.json"
        inspec_records = []
        for index, color in enumerate(("red", "blue"), start=1):
            filename = f"test__site-Level01-{index:04d}__frame-{index}.jpg"
            self._image(inspec_root / filename, color)
            inspec_records.append(
                {
                    "id": f"ins-{index}",
                    "image": f"images/{filename}",
                    "messages": [
                        {"role": "system", "content": "Preserve this prompt."},
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "hazards": ["smoke"] if index == 1 else [],
                                    "overall_safety_level": (
                                        "Level I" if index == 1 else "Level IV"
                                    ),
                                }
                            ),
                        }
                    ],
                    "metadata": {"split": "test", "source_index": index},
                }
            )
        # Exact-ID exports must not fail strict validation because an unrelated
        # source annotation points at an unavailable image.
        inspec_records.append(
            {
                "id": "ins-unrelated-missing",
                "image": "images/test__site-Level01-missing__missing.jpg",
                "messages": [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "hazards": [],
                                "overall_safety_level": "Level IV",
                            }
                        ),
                    }
                ],
                "metadata": {"split": "test"},
            }
        )
        inspec_annotations.write_text(
            json.dumps(inspec_records),
            encoding="utf-8",
        )

        labsafety_root = root / "labsafety"
        labsafety_annotations = labsafety_root / "annotations.jsonl"
        labsafety_root.mkdir(parents=True)
        lab_records = []
        for index, color in enumerate(("green", "yellow"), start=1):
            relative = Path("images") / "test" / f"lab-{index}.png"
            self._image(labsafety_root / relative, color)
            lab_records.append(
                {
                    "image_id": f"lab-{index}",
                    "image": relative.as_posix(),
                    "split": "test",
                    "safety_label": "hazardous" if index == 1 else "non-hazardous",
                    "hazards": ["spill"] if index == 1 else [],
                    "description": f"Lab sample {index}",
                    "vlm_label": "hazardous",
                    "agree": index == 1,
                    "generator": "fixture-v1",
                }
            )
        labsafety_annotations.write_text(
            "".join(json.dumps(record) + "\n" for record in lab_records),
            encoding="utf-8",
        )
        return (
            inspec_annotations,
            inspec_root,
            labsafety_annotations,
            labsafety_root,
        )

    def test_exports_exact_entries_from_both_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                inspec_annotations,
                inspec_root,
                labsafety_annotations,
                labsafety_root,
            ) = self._source_datasets(root)
            output = root / "portable_data"

            manifest = export_subsets(
                output_dir=output,
                datasets=[
                    INSPECSAFE_SAFETY_LEVEL_DATASET,
                    LABSAFETY_GEN_DATASET,
                ],
                annotations_by_dataset={
                    INSPECSAFE_SAFETY_LEVEL_DATASET: inspec_annotations,
                    LABSAFETY_GEN_DATASET: labsafety_annotations,
                },
                image_roots_by_dataset={
                    INSPECSAFE_SAFETY_LEVEL_DATASET: inspec_root,
                    LABSAFETY_GEN_DATASET: labsafety_root,
                },
                ids_by_dataset={
                    INSPECSAFE_SAFETY_LEVEL_DATASET: ["ins-2"],
                    LABSAFETY_GEN_DATASET: ["lab-1"],
                },
                split="test",
                offset=0,
                limit=None,
                shuffle=False,
                seed=0,
                strict_images=True,
            )

            self.assertTrue((output / "manifest.json").is_file())
            self.assertEqual(
                manifest["datasets"][INSPECSAFE_SAFETY_LEVEL_DATASET]["count"],
                1,
            )
            self.assertEqual(
                manifest["datasets"][LABSAFETY_GEN_DATASET]["items"][0]["sample_id"],
                "lab-1",
            )

            exported_inspec, missing_inspec = load_display_samples(
                dataset=INSPECSAFE_SAFETY_LEVEL_DATASET,
                annotations_path=(
                    output / INSPECSAFE_SAFETY_LEVEL_DATASET / "annotations.json"
                ),
                image_root=output / INSPECSAFE_SAFETY_LEVEL_DATASET,
                split="all",
                skip_missing=False,
            )
            exported_lab, missing_lab = load_display_samples(
                dataset=LABSAFETY_GEN_DATASET,
                annotations_path=output / LABSAFETY_GEN_DATASET / "annotations.jsonl",
                image_root=output / LABSAFETY_GEN_DATASET,
                split="all",
                skip_missing=False,
            )
            self.assertEqual((missing_inspec, missing_lab), (0, 0))
            self.assertEqual(exported_inspec[0].sample_id, "ins-2")
            self.assertEqual(
                exported_inspec[0].ground_truth["overall_safety_level"],
                "Level IV",
            )
            self.assertEqual(exported_lab[0].sample_id, "lab-1")
            self.assertEqual(exported_lab[0].ground_truth["safety_label"], "hazardous")
            raw_inspec = json.loads(
                (
                    output / INSPECSAFE_SAFETY_LEVEL_DATASET / "annotations.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(raw_inspec[0]["messages"][0]["role"], "system")
            raw_lab = json.loads(
                (output / LABSAFETY_GEN_DATASET / "annotations.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(raw_lab["generator"], "fixture-v1")

    def test_refuses_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export_subsets(
                    output_dir=output,
                    datasets=[],
                    annotations_by_dataset={},
                    image_roots_by_dataset={},
                    ids_by_dataset={},
                    split="test",
                    offset=0,
                    limit=None,
                    shuffle=False,
                    seed=0,
                    strict_images=False,
                )

    def test_positional_selection_is_reproducible(self) -> None:
        samples = [
            DisplaySample(
                dataset=LABSAFETY_GEN_DATASET,
                sample_id=f"sample-{index}",
                image_path=Path(f"{index}.png"),
                source_image=f"{index}.png",
                ground_truth={},
                metadata={},
            )
            for index in range(8)
        ]
        first = select_samples(
            samples,
            requested_ids=[],
            offset=1,
            limit=3,
            shuffle=True,
            seed=42,
        )
        second = select_samples(
            samples,
            requested_ids=[],
            offset=1,
            limit=3,
            shuffle=True,
            seed=42,
        )
        self.assertEqual(
            [sample.sample_id for sample in first],
            [sample.sample_id for sample in second],
        )


if __name__ == "__main__":
    unittest.main()
