# Offline near-range structured-light supplement

This directory reconstructs additional sparse near-range depth from the original, synchronous left/right IR images. It is CPU postprocessing of a sealed native capture, not a change to the original Isaac Sim sensor configuration. Native RGB, GT depth, IR, tactile, calibration, timestamps and original SL remain in the original HDF5.

## Fixed reconstruction and limits

The native 50 mm baseline and 128 px search can miss wrist surfaces around 6 cm. The fixed candidate searches 384/448/512 px on the same IR pair with the original bidirectional consistency, texture and patch-correlation gates. These searches are not statistically independent measurements. Additional samples must agree within 0.5 px, have disparity at least 128 px, remain in both real eye domains, and pass a separate 9×9 IR correlation/support check. The support threshold is 90%, not a fully valid patch. Zero padding affects optimizer context but does not provide replacement image evidence or synthetic correspondences.

Only original NaN values can receive new finite measurements. Every original finite value is kept bitwise, including any original mismatches. The chosen additions come from the 384 px result, so this does not claim 512 px output coverage. Occlusion and weak correspondence remain NaN. GT, actions, contact state and task stage are never reconstruction inputs. RGB GT depth and reconstructed SL use different optical references; accuracy evaluation must account for calibration and mutual visibility. Neither completed storage nor lineage validation certifies dense depth or whole-episode accuracy.

The exact independently reviewed candidate file is frozen at SHA256 `291fdeb748be893e3ceaf513c37770540dcc1eeb8109148ed61c5a9a484906fd`; its native reconstruction dependency is frozen at `523087ace2a9ae291a089a5b6b0e9be6ed17b40c83f1d367fff91bcde37fea10`. Import checks both bytes before candidate execution. `derive.py` alone is excluded from formatting and its actual E501/E701/E702/E741/I001 findings are narrowly exempted to preserve these verified bytes. All other Python files follow repository Ruff rules.

## Derive or resume from a completed recording

Run from the RoboDojo repository root with the configured CPU Python environment:

```bash
.venv-isaac61/bin/python reproduction/isaac61/sl_near_range/resume_sidecar.py \
  --final-source reproduction/isaac61/outputs/insert-tubes/full-episode-final/episode.hdf5 \
  --output reproduction/isaac61/outputs/insert-tubes/structured-light-derived/episode.derived-sl.hdf5
```

This creates a sidecar, or resumes its existing `.partial` prefix. It checks complete source metadata and hashes the whole sealed source, verifies every old frame's IR/original SL/calibration/provenance/physical clock and derived payload hashes, then processes only additional frames. The original finite SL bits and exact added mask are checked. Completion requires all source frames and unchanged source size/mtime; only then is the sidecar renamed to `.hdf5`. Existing completed outputs are refused. No source HDF5 is modified.

If an explicit immutable COPY was produced while capture was quiescent, prefix processing can run with `--stable-copy COPY.hdf5.partial --copy-receipt COPY.json` instead of `--final-source`. The receipt must identify the original active path, verified stability, count and whole copy SHA. Prefix results stay `.partial/status=incomplete` and are not publishable as complete. Hardlinks, symlink aliases and colliding output paths to the original active object are rejected before that object can be hashed or opened. Never supply the live recording itself.

Sidecar schema `robodojo.structured_light_derived.hdf5.v1` preserves original and derived SL plus `added_mask` for each camera/frame; IR hashes, calibration, provenance, clocks, source identities, fixed parameters and algorithm pin are stored for audit. `complete` means this sidecar covers the sealed source, and does not claim task success. Native task acceptance stays in its own recording receipt.

## Supplemental PNG and MP4

Use a fresh output directory, after both input files are complete:

```bash
.venv-isaac61/bin/python reproduction/isaac61/sl_near_range/visualize_derived.py \
  --source reproduction/isaac61/outputs/insert-tubes/full-episode-final/episode.hdf5 \
  --sidecar reproduction/isaac61/outputs/insert-tubes/structured-light-derived/episode.derived-sl.hdf5 \
  --output reproduction/isaac61/outputs/insert-tubes/structured-light-derived/visualization-new \
  --preview-frame 296 --depth-min 0.04 --depth-max 2.5
```

The viewer checks every frame before encoding, including frames not displayed when using `--frame-stride`. It rejects partial files, incomplete metadata, source/hash/clock mismatches and existing named artifacts. It only substitutes in-memory `depth_sl` for the canonical mosaic. RGB, GT, IR and tactile are unchanged. The banner and each SL tile say **near-range derived SL**. Purple means NaN; fixed display color limits and physics-clock FPS are recorded. Both input hashes are written to the output JSON. This supplemental video does not replace the original native capture exports. A failed export leaves no success report, and its output directory must not be reused.

## CPU contracts

```bash
.venv-isaac61/bin/python reproduction/isaac61/sl_near_range/test_derive.py -v
.venv-isaac61/bin/python reproduction/isaac61/sl_near_range/test_resume.py -v
.venv-isaac61/bin/python reproduction/isaac61/sl_near_range/test_visualize.py -v
.venv-isaac61/bin/ruff check reproduction/isaac61/sl_near_range
```

Tests use temporary synthetic typed-HDF5 fixtures and real CPU ffmpeg export. They cover known near disparity, negative mismatched/occluded/untextured eyes, original finite-bit preservation, resumed prefix corruption, frozen source pins, active hardlink/symlink/output aliases, incomplete input, skipped-frame corruption and stale output refusal. Native image commissioning and native task acceptance are separate archived evidence.
