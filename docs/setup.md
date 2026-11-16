# Setting up an existing simulator environment

1. Install the upstream simulator and datasets using their instructions. Verify
   that their normal single-environment example works before adding this adapter.
2. Use a checkout matching the compatibility profile:
   [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K/tree/26f2c7ef7b9cf96bd0414f81e1e751e493762779),
   [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c),
   and RoboDojo's pinned Isaac Lab submodule. SDK versions are in
   [compatibility.json](../src/kit_timeshare/compatibility.json).
3. Install this package using that environment's Python. You can also install a
   release wheel offline with `python -m pip install --no-deps <wheel>`.
4. Run `kit-timeshare doctor` and resolve version mismatches. The command reads
   device information and does not start a simulation.
5. Isolate one physical GPU **and** its corresponding DRM render device. On a
   multi-GPU host, use an NVIDIA Container Toolkit container based on your own
   working simulator image. `docker-command` prints a starting command; add the
   data mounts and environment settings your upstream installation normally needs.
6. Run `examples/native_acceptance.py` for your backend before connecting a
   long-running policy. Choose a fresh output directory and keep the report.

```bash
# Run inside the isolated SDK environment, from this repository checkout.
python examples/native_acceptance.py behavior \
  --upstream-root /path/to/BEHAVIOR-1K --output-dir /path/to/new-acceptance-run

python examples/native_acceptance.py robodojo \
  --upstream-root /path/to/RoboDojo --output-dir /path/to/another-new-run
```

The default BEHAVIOR YAML is OmniGibson's `r1pro_behavior.yaml`; pass
`--config-path` to validate your own fixed-topology task configuration. RoboDojo's
reference profile is `stack_bowls`, dual ARX X5, and its saved evaluation layout.
Using another task is an additional compatibility experiment, not a certified
benchmark evaluation.

Kit can compile shaders and resolve missing online material dependencies during
startup. Startup time is not the per-step switching cost. The adapter does not
provide a simulator Docker image or redistribute datasets; the small Python
wheel alone cannot run a licensed simulator.
