"""Model-free tests for the two-stage InspecSafe decision policy."""

import unittest

from two_stage_inference import run_two_stage_safety_inference


class StubGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[object, str, int]] = []

    def __call__(self, image: object, prompt: str, limit: int) -> str:
        self.calls.append((image, prompt, limit))
        return next(self.outputs)


class TwoStageSafetyInferenceTests(unittest.TestCase):
    def test_safe_first_stage_skips_second_generation(self) -> None:
        generator = StubGenerator(["safe"])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertIsNone(result["stage_two"])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0][2], 8)

    def test_second_safe_result_overrides_first_unsafe_result(self) -> None:
        generator = StubGenerator([
            "unsafe",
            "Annotation: A possible issue is visible.\nFinal label: safe",
        ])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(generator.calls[1][2], 128)

    def test_two_unsafe_results_return_annotation(self) -> None:
        generator = StubGenerator([
            "unsafe",
            "Annotation: Worker is missing a hard hat.\nFinal label: unsafe",
        ])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["output"], "unsafe")
        self.assertEqual(result["label"], "unsafe")
        self.assertEqual(result["annotation"], "Worker is missing a hard hat.")

    def test_unparseable_second_stage_becomes_safe(self) -> None:
        generator = StubGenerator(["unsafe", "The image should be reviewed."])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertIsNone(result["stage_two"]["label"])

    def test_ambiguous_first_stage_does_not_trigger_second_generation(self) -> None:
        generator = StubGenerator(["safe or unsafe"])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertIsNone(result["stage_one"]["label"])
        self.assertEqual(len(generator.calls), 1)


if __name__ == "__main__":
    unittest.main()
