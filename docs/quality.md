# Compare recorded sensor quality with original references

`scripts/build_sensor_quality_report.py` builds an offline report from saved
runtime data. It starts no Kit process and performs no GPU simulation or image
synthesis. Run it with the Python environment used for HDF5 capture:

```bash
.venv-isaac61/bin/python scripts/build_sensor_quality_report.py \
  --reference-dir reproduction/isaac61/outputs/quality-iteration/review-reference \
  --capture-dir reproduction/isaac61/outputs/quality-iteration/fullmodal-contact \
  --tactile-dir reproduction/isaac61/outputs/tactile-nut-contact-61 \
  --output reproduction/isaac61/outputs/quality-iteration/report
```

Use the actual completed capture directories. The example paths describe the
intended inputs, and do not assert that a capture exists or passed. The default
`--baseline-dir` is the earlier `moving-wrists-61-v5-final` NPZ recording. Its
measured zero contact force is identified as a noncontact baseline. A new HDF5
capture is not relabelled with an old baseline source revision.

Optional `--sl-dir` arguments add the separately audited planar, tilted or frozen
RGB controls, for example `--sl-dir reproduction/isaac61/outputs/quality-iteration/sl-reference-plane-v3`.
Repeat the flag for more controls. Their original `report.json` and independent
status remain separate from the main HDF5 capture. A passed control audit must
link `source_report_sha256` and every original NPZ in `source_files` through its
SHA256. Controls never substitute for the required capture and contact sequence.
The reader is bounded to 32 NPZ files / 1 GiB per control directory.

The output is `index.html` and `report.json`. PNGs and original JSON provenance
are embedded in the page. MP4 and saved viewers use local links; add
`--copy-videos` to copy MP4 files into the report directory. The referenced
HDF5/NPZ files remain in their original directories. The report retains every
loaded JSON input's path and SHA256, along with original statuses, versions,
source/configuration hashes and limitations. Dark mode and measurement filtering
work offline. Full source JSON is also available in the page's provenance panel.

A new multimodal HDF5 capture must have `manifest.json` (`storage: "hdf5"`,
`file: "episode.hdf5"`, `complete: true`, indexed frame inventory), an explicitly
passed real-runtime `validation.json`, and an explicitly passed independent
frame audit linked to `source_hdf5` and `source_hdf5_sha256`. The builder rereads
all committed HDF5 frames through `HDF5Reader` / `validate_hdf5`, checks sensor
shapes and synchronization, and compares actual clocks and frame counts with the
original manifest and validation. It recomputes the source SHA256. A component
TacSL contact sequence uses `robodojo.tactile_component.v1`; its schema validator
checks the four real pads and clocks without requiring fake camera payloads. Its
original report and independent audit must still name and hash that actual HDF5
source. A component sequence is never accepted as a full three-camera capture.

Runtime/audit statuses must be explicit and consistent. Missing independent
acceptance produces an `incomplete` report; failed acceptance remains `failed`.
Malformed reports, conflicting frame inventories, invalid clocks and modified
source hashes are rejected. `--require-passed` writes the truthful report and
returns exit code 1 unless all supplied evidence passes. Reference videos are
labelled qualitative comparisons, never acceptance tests; their extraction
manifest must retain and match both video and frame hashes.

The HDF5 viewer's `visualization.json` links the preview to its actual episode
and records the selected frame and tactile score. The quality builder checks
that it is the strongest scored frame in the committed sequence, rather than an
idle frame before contact. Use the viewer's full default frame inventory for a
final quality report, without frame stride or a truncated maximum frame count. An optical-only fallback must state that no force claim
is made. Selection and display scaling remain visible beside the preview;
amplified rendering is not presented as raw calibrated RGB. The builder does
not create or alter sensor images.

IR is the current RTX visible-radiance proxy rather than full NIR material
response. TacSL is IsaacLab's optical/penalty model, not UniVTAC UIPC or a marker
tracking simulation. Reported raw force uses newtons, depth/penetration uses
metres, and fractions use 0–1. Differences between reference scenes do not imply
a measured improvement percentage or policy success.

The existing `scripts/build_acceptance_report.py` also accepts these HDF5
recordings for its `--moving-dir`, `--fluid-dir` and `--cloth-dir` inputs while
retaining legacy NPZ compatibility and per-run source metadata.
