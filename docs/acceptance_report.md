# Build a portable acceptance report

`scripts/build_acceptance_report.py` reads existing results and starts no simulator.
Run it from the RoboDojo checkout. For example, this command reproduces a report
from the initial task-suite run, including its failures:

```bash
bash reproduction/isaac61/run.sh python scripts/build_acceptance_report.py \
  --suite-dir reproduction/isaac61/outputs/task-suite-final \
  --cpu-log reproduction/isaac61/outputs/review/target61-cpu-tests-final.log \
  --moving-dir reproduction/isaac61/outputs/moving-wrists-61-final \
  --fluid reproduction/isaac61/outputs/fluid-multimodal-61 \
  --cloth reproduction/isaac61/outputs/cloth-multimodal-61 \
  --tactile reproduction/isaac61/outputs/tactile-contact-readback-env2 \
  --sl reproduction/isaac61/outputs/sl-tilted-plane-61 \
  --replay reproduction/isaac61/outputs/bowl-replay \
  --physics-gallery reproduction/isaac61/outputs/physics-gallery \
  --output reproduction/isaac61/outputs/acceptance-report
```

Select a new suite directory and CPU log for later runs. Optional evidence
arguments can be omitted; explicitly supplied directories must exist and contain
their expected reports and previews. `--fluid-dir`, `--cloth-dir`, `--tactile-dir`,
`--sl-dir`, `--replay-dir` and `--physics-gallery-dir` are equivalent longer forms.

Add `--require-passed` for final acceptance. The builder still writes a truthful
report for a failed suite, but then exits nonzero if the suite, CPU tests or any
supplied runtime evidence did not pass. It rejects inconsistent suite counts,
incomplete recordings claimed as passed, missing frame files and nonportable
viewer dependencies. Creating a report without this option does not establish
that acceptance passed; inspect its recorded status.

The output contains `index.html`, `report.json` and copies of the supplied HTML
viewers under `viewers/`. The main page embeds PNG previews and interactive
viewers, so it also works as a standalone offline HTML file. `report.json` retains
the original suite, individual task statuses, completeness, source commit,
packages and per-run results. Input paths and SHA256 hashes identify the evidence.
CPU test counts come from the last completed pytest summary in the selected log.
Different source revisions remain visible rather than being relabelled with the
current checkout's revision.

Raw NPZ recordings stay in their original directories; the report package does
not duplicate them. No policy success rate is inferred. The page explicitly
records the visible-radiance IR approximation, changed FEM cloth physics,
rigid-collider scope of tactile forces and the possibility of zero force in
noncontact recordings. Cloth-only unsupported force fields remain null in the
original evidence.
