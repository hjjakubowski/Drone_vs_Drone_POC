import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from project.config import load_config
from project.vision.engine_metadata import check_engine_metadata, read_engine_metadata
from scripts.export_model import main


class EngineMetadataTests(unittest.TestCase):
    def setUp(self):
        self.options = load_config("model")["export"]
        self.metadata = {
            "batch": 1, "imgsz": [544, 960], "task": "detect",
            "names": {"2": "drone"}, "end2end": True,
            "args": {"quantize": 16, "dynamic": False, "nms": False},
        }

    def test_rectangular_end_to_end_metadata_matches(self):
        self.assertEqual(check_engine_metadata(self.metadata, self.options), [])

    def test_square_raw_output_export_is_rejected_by_inspection(self):
        metadata = copy.deepcopy(self.metadata)
        metadata.update(imgsz=[640, 640], end2end=False)
        metadata["args"]["nms"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.engine"
            encoded = json.dumps(metadata).encode()
            path.write_bytes(len(encoded).to_bytes(4, "little") + encoded + b"engine data")
            self.assertEqual(read_engine_metadata(path), metadata)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--inspect", str(path)]), 1)
            report = json.loads(output.getvalue())
            self.assertEqual(len(report["differences"]), 3)
            self.assertFalse(report["gpu_verified"])

    def test_truncated_and_non_ultralytics_files_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.engine"
            for content in (b"", b"abcd", b"\x10\x00\x00\x00{}", b"\x02\x00\x00\x00{}"):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    read_engine_metadata(path)


if __name__ == "__main__":
    unittest.main()
