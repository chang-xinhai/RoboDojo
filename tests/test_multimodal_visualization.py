"""Viewer display and HDF5-only export checks on declared CPU fixtures."""

from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.convert_multimodal_hdf5 import convert
from reproduction.visualize_multimodal import _taxels, letterbox, metric_color, mosaic, visualize
from tests.test_multimodal import fixture
from utils.multimodal import MultimodalWriter
from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer


class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.inside = False
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            if dict(attrs).get("src"):
                raise AssertionError("Standalone viewer must not load external scripts")
            self.inside, self.buffer = True, []

    def handle_data(self, value):
        if self.inside:
            self.buffer.append(value)

    def handle_endtag(self, tag):
        if tag == "script":
            self.scripts.append("".join(self.buffer))
            self.inside = False


NODE_VIEWER_CHECK = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const scripts = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const maximumStringLength = require('node:buffer').constants.MAX_STRING_LENGTH;
const ids = ['slider', 'mode', 'picture', 'counter', 'details', 'play'];
const elements = Object.fromEntries(ids.map(id => [id, {value: '0'}]));
let tick, cancelled = false;
const sandbox = {document: {getElementById: id => elements[id]},
  setInterval: callback => {tick = callback; return 1},
  clearInterval: () => {cancelled = true}};
vm.createContext(sandbox);
for (const script of scripts) {
  assert(script.length < maximumStringLength);
  vm.runInContext(script, sandbox);
}
assert.equal(vm.runInContext('frames.length', sandbox), 3);
assert.equal(elements.slider.max, 2);
assert.equal(Number(elements.slider.value), 1);
assert.equal(elements.mode.value, 'rgb');
assert.equal(JSON.parse(elements.details.textContent).sync.physics_frame, 2);
assert.equal(elements.picture.src, vm.runInContext('frames[1].views.rgb', sandbox));
const modes = ['rgb', 'optical_depth_delta', 'normal_force', 'shear_force', 'pad_normal', 'pad_shear',
  'raw_optical', 'raw_optical_depth_delta'];
for (const mode of modes) {
  elements.mode.value = mode;
  elements.mode.onchange();
  assert.equal(elements.picture.src, vm.runInContext('frames[1].views[' + JSON.stringify(mode) + ']', sandbox));
}
elements.slider.value = '2';
elements.slider.oninput();
assert.equal(JSON.parse(elements.details.textContent).sync.physics_frame, 3);
assert.equal(elements.counter.textContent, '3 / 3');
elements.play.onclick();
tick();
assert.equal(Number(elements.slider.value), 0);
assert.equal(JSON.parse(elements.details.textContent).sync.physics_frame, 1);
elements.play.onclick();
assert.equal(cancelled, true);
assert.equal(vm.runInContext('typeof alert', sandbox), 'undefined');
console.log(JSON.stringify({status: 'passed', frames: 3, modes: modes.length, scripts: scripts.length,
  maximumScriptChars: Math.max(...scripts.map(script => script.length)), maximumStringLength,
  slider: true, play: true}));
"""


class VisualizationTests(unittest.TestCase):
    def test_letterbox_preserves_portrait_geometry(self):
        image = np.full((320, 240, 3), 255, np.uint8)
        rendered = np.asarray(letterbox(image, (280, 160)))
        white = np.all(rendered == 255, axis=-1)
        self.assertEqual(white.sum(), 120 * 160)
        self.assertEqual(white[:, :80].sum(), 0)
        self.assertTrue(white[:, 80:200].all())

    def test_declared_official_taxel_axes_display_z_rows_and_x_columns(self):
        data = {
            "normal_force": np.arange(6, dtype=np.float32),
            "metadata": {"array_size": [2, 3], "taxel_array_axes": ["sensor_local_x", "sensor_local_z"]},
        }
        np.testing.assert_equal(_taxels(data, "normal_force"), [[0, 3], [1, 4], [2, 5]])
        del data["metadata"]["taxel_array_axes"]
        np.testing.assert_equal(_taxels(data, "normal_force"), [[0, 1, 2], [3, 4, 5]])

    def test_metric_scale_fixed_and_invalid_distinct(self):
        a = metric_color(np.array([[1.0, 1.5], [np.nan, 2.0]]), 0.1, 2.5)
        b = metric_color(np.array([[0.0, 1.0], [2.5, 20.0]]), 0.1, 2.5)
        np.testing.assert_equal(a[0, 0], b[0, 1])
        np.testing.assert_equal(a[1, 0], [130, 0, 130])

    def test_direct_hdf5_no_npz_selects_measured_contact_and_exports_video(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(directory, {"synthetic_cpu_fixture": True}) as writer:
                for index in range(1, 4):
                    frame = fixture(index)
                    sample = frame["tactile"]["left"]["finger_0"]
                    sample["normal_force"][0] = 0.2 if index == 2 else 0
                    sample["optical_depth_delta"] = np.full((2, 4, 1), -0.001, np.float32)
                    sample["metadata"] = {"array_size": [2, 4]}
                    frame["vision"]["head"]["provenance"]["untrusted"] = "</script><script>alert(1)</script>"
                    writer.append(frame)
            report = visualize(writer.path)
            self.assertEqual(report["source_format"], "hdf5")
            self.assertEqual(report["selected_frame"]["physics_frame"], 2)
            self.assertEqual(report["frames"], 3)
            self.assertFalse(list(directory.glob("*.npz")))
            self.assertTrue(Path(report["video"]).stat().st_size > 100)
            html = Path(report["html"]).read_text()
            self.assertIn("normal_force", html)
            self.assertIn("const selectedPreviewIndex=1", html)
            self.assertNotIn("</script><script>alert", html)
            self.assertIn("\\u003c/script", html)
            self.assertEqual(json.loads((directory / "visualization.json").read_text())["selected_frame"]["index"], 1)
            parser = ScriptCollector()
            parser.feed(html)
            self.assertEqual(len(parser.scripts), 5)  # init + three independent frames + controls
            self.assertEqual(parser.scripts[0], "const frames=[];")
            blocks = parser.scripts[1:-1]
            for index, block in enumerate(blocks):
                self.assertTrue(block.startswith("frames.push("))
                self.assertTrue(block.endswith(");"))
                payload = json.loads(block[len("frames.push(") : -len(");")])
                self.assertEqual(payload["details"]["sync"]["physics_frame"], index + 1)
                self.assertEqual(
                    set(payload["views"]),
                    {
                        "rgb",
                        "optical_depth_delta",
                        "normal_force",
                        "shear_force",
                        "pad_normal",
                        "pad_shear",
                        "raw_optical",
                        "raw_optical_depth_delta",
                    },
                )
                self.assertEqual(
                    payload["details"]["vision"]["head"]["provenance"]["untrusted"],
                    "</script><script>alert(1)</script>",
                )
                self.assertLess(len(block), len(html) / 2)
            self.assertNotIn("const frames=[\n", html)
            if node := shutil.which("node"):
                scripts_file = directory / "scripts.json"
                scripts_file.write_text(json.dumps(parser.scripts))
                result = subprocess.run(
                    [node, "-e", NODE_VIEWER_CHECK, str(scripts_file)], capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                runtime = json.loads(result.stdout)
                self.assertEqual(runtime["scripts"], 5)
                self.assertTrue(runtime["slider"] and runtime["play"])

    def test_legacy_conversion_preserves_original_runtime_evidence_and_null_forces(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MultimodalWriter(directory, {"source_commit": "original-fixture", "real_runtime": False})
            frame = fixture()
            for samples in frame["tactile"].values():
                for sample in samples.values():
                    for field in ("normal_force", "shear_force", "penetration_depth", "points_pos_w"):
                        sample[field] = None
                    sample["provenance"].update(force_field_available=False, force_unavailable_reason="fixture cloth")
            writer.append(frame)
            writer.close()
            report = convert(directory)
            with HDF5Reader(report["file"]) as reader:
                self.assertIsNone(reader.read_frame(0)["tactile"]["left"]["finger_0"]["shear_force"])
                metadata = reader.metadata["run"]
                self.assertEqual(metadata["source_commit"], "original-fixture")
                self.assertTrue(metadata["format_conversion"]["conversion_only"])
                self.assertTrue(metadata["format_conversion"]["original_files"]["frame_000000.npz"])
            self.assertTrue((Path(directory) / "frame_000000.npz").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Video tools required")
    def test_video_only_preserves_all_frames_timing_and_contact_without_html_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(directory / "source", {"synthetic_cpu_fixture": True}) as writer:
                for index in range(1, 6):
                    frame = fixture(index)
                    frame["tactile"]["right"]["finger_1"]["normal_force"][0] = 0.3 if index == 4 else 0
                    writer.append(frame)
            output = directory / "video-only"
            with patch("reproduction.visualize_multimodal.mosaic", wraps=mosaic) as renderer:
                report = visualize(writer.path, output, include_html=False, tactile="normal_force")
            self.assertEqual(renderer.call_count, 5)  # Only the requested mode; no eight-mode HTML encoding.
            self.assertEqual(report["frames"], 5)
            self.assertEqual(report["selected_frame"]["index"], 3)
            self.assertEqual(report["selected_frame"]["physics_frame"], 4)
            self.assertEqual(report["tactile_display_mode"], "normal_force")
            self.assertIsNone(report["html"])
            self.assertFalse((output / "multimodal.html").exists())
            self.assertTrue(Path(report["png"]).is_file())
            self.assertEqual(json.loads((output / "visualization.json").read_text()), report)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                    report["video"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(int(stream["nb_read_frames"]), 5)
            self.assertEqual(stream["r_frame_rate"], "25/1")
            self.assertAlmostEqual(float(stream["duration"]), 0.2)
