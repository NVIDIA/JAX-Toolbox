"""GPU-free tests: flags, CLI plans, comparisons, and mocked execution flows."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment as runner
import compare_results as compare


class PlanTests(unittest.TestCase):
    def plan(self, model="gemma", mode="benchmark", env=None, extra=()):
        args = runner.parse_args([model, mode, "--out-dir", "/tmp/pdl-test-plan", *extra])
        return runner.make_plan(args, env or {})

    def test_all_models_and_modes(self):
        for model in ("gemma", "timesfm", "alphafold"):
            for mode in ("benchmark", "xprof", "nsys"):
                with self.subTest(model=model, mode=mode):
                    plan = self.plan(model, mode)
                    self.assertEqual(len(plan["jobs"]), 2)
                    self.assertEqual(plan["profiled"], mode != "benchmark")
                    for job in plan["jobs"]:
                        self.assertTrue(job["graphs"])
                        self.assertIn(f"--xla_gpu_enable_pdl={str(job['pdl']).lower()}", job["env"]["XLA_FLAGS"])
                        self.assertIn(f"--xla_gpu_enable_pdl_launch={str(job['pdl']).lower()}", job["env"]["XLA_FLAGS"])
                        self.assertEqual(job["env"]["JAX_ENABLE_COMPILATION_CACHE"], "false")

    def test_preserve_flags_and_replace_pdl(self):
        env = {"XLA_FLAGS": '--unrelated=123 --xla_gpu_enable_pdl=true --xla_dump_to="/a path/dump"',
               "XLA_FLAGS_EXTRA": "--xla_gpu_enable_pdl_launch true"}
        flags = runner.arm_flags(env, "timesfm", True, False)
        self.assertTrue(flags.startswith("--"))
        self.assertIn('--xla_dump_to="/a path/dump"', flags)
        tokens = shlex.split(flags)
        self.assertIn("--unrelated=123", tokens)
        self.assertEqual(tokens.count("--xla_gpu_enable_pdl=false"), 1)
        self.assertNotIn("--xla_gpu_enable_pdl=true", tokens)

    def test_empty_command_buffers_rejected(self):
        for spelling in ("--xla_gpu_enable_command_buffer=", "--xla_gpu_enable_command_buffer ''"):
            with self.assertRaisesRegex(ValueError, "disables graphs"):
                self.plan(env={"XLA_FLAGS": spelling})

    def test_nonempty_command_buffers_preserved(self):
        plan = self.plan(env={"XLA_FLAGS": "--xla_gpu_enable_command_buffer=FUSION,CUBLAS"})
        self.assertIn("--xla_gpu_enable_command_buffer=FUSION,CUBLAS", plan["jobs"][0]["env"]["XLA_FLAGS"])

    def test_matrix_and_alternating_order(self):
        plan = self.plan(extra=("--matrix", "--repeats", "2"))
        self.assertEqual(len(plan["jobs"]), 8)
        self.assertEqual([j["pdl"] for j in plan["jobs"][:2]], [False, True])
        self.assertEqual([j["pdl"] for j in plan["jobs"][4:6]], [True, False])
        for job in plan["jobs"]:
            self.assertEqual("--xla_gpu_enable_command_buffer=" in shlex.split(job["env"]["XLA_FLAGS"]), not job["graphs"])

    def test_tp4_exactly_four_devices(self):
        plan = self.plan(env={"TP": "4", "PER_DEVICE_BS": "1"})
        self.assertEqual(plan["jobs"][0]["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        with self.assertRaises(ValueError):
            self.plan(env={"TP": "4", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4"})
        with self.assertRaises(ValueError):
            self.plan(env={"CUDA_VISIBLE_DEVICES": "-1"})

    def test_nsys_jax_separator_and_graph_nodes(self):
        command = self.plan(mode="nsys", extra=("--nsys-tool", "nsys-jax"))["jobs"][0]["command"]
        self.assertEqual(command[0], "nsys-jax")
        self.assertIn("--", command)
        self.assertIn("--cuda-graph-trace=node", command)
        self.assertIn("--python-sampling=false", command)
        self.assertFalse(any("force-overwrite" in arg for arg in command))

    def test_profile_shapes_match_benchmark(self):
        for model in ("gemma", "timesfm", "alphafold"):
            baseline = self.plan(model)["settings"]
            for mode in ("xprof", "nsys"):
                settings = self.plan(model, mode)["settings"]
                self.assertEqual({k: v for k, v in settings.items() if k != "PROFILE_ITERS"},
                                 {k: v for k, v in baseline.items() if k != "PROFILE_ITERS"})

    def test_gemma_native_profiler_switch(self):
        benchmark = self.plan()["jobs"][0]["command"]
        self.assertIn("profiler=''", benchmark)
        self.assertNotIn("load_parameters_path=", benchmark)
        capture = self.plan(mode="xprof")["jobs"][0]["command"]
        self.assertIn("profiler=xplane", capture)
        self.assertIn("inference_microbenchmark_prefill_lengths=1024,1024", capture)

    def test_invalid_shapes_and_ambiguous_weights(self):
        for model, env in (("timesfm", {"HORIZON_LEN": "129"}),
                           ("timesfm", {"LEVEL": "e2e", "DTYPE": "bfloat16"}),
                           ("gemma", {"RANDOM_WEIGHTS": "0"}),
                           ("gemma", {"CKPT": "/weights"}),
                           ("alphafold", {"WARM_RUNS": "0"})):
            with self.subTest(model=model, env=env), self.assertRaises(ValueError):
                self.plan(model=model, env=env)

    def test_every_shell_entrypoint_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ("gemma", "timesfm", "alphafold"):
                for mode in ("benchmark", "xprof", "nsys"):
                    wrapper = ROOT / model / f"run_{mode}.sh"
                    subprocess.run(["bash", "-n", str(wrapper)], check=True)
                    out = Path(tmp) / f"{model}-{mode}"
                    result = subprocess.run(["bash", str(wrapper), "--dry-run", "--out-dir", str(out)],
                                            env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable},
                                            capture_output=True, text=True, check=True)
                    self.assertEqual(json.loads(result.stdout)["model"], model)
                    self.assertFalse(out.exists())


class CompareTests(unittest.TestCase):
    def fixture(self, directory, model="timesfm", mode="benchmark"):
        plan = runner.make_plan(runner.parse_args([model, mode, "--out-dir", str(directory)]), {})
        for job in plan["jobs"]:
            path = Path(job["raw_json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            job["status"] = "complete"
            value = 8 if job["pdl"] else 10
            if model == "gemma":
                raw = {"autoregressive": {"step_in_ms": value, "global_batch_size": 4}}
            elif model == "alphafold":
                raw = {"warm_execute_times_s": [value, value], "tokens_used": 2048}
            else:
                raw = {"samples_ms": [value, value], "aot_compiled": True, "horizon_len": 512}
            path.write_text(json.dumps(raw))
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(plan))
        return manifest

    def test_units_and_loop_mean_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ("gemma", "timesfm", "alphafold"):
                manifest = self.fixture(Path(tmp) / model, model)
                summary = compare.summarize(manifest)
                self.assertEqual(summary["marginal_pdl_reduction_pct"], 20)
                self.assertEqual(summary["pairs"][0]["off_ms"], 10000 if model == "alphafold" else 10)
                self.assertEqual(summary["records"][0]["sample_count"], None if model == "gemma" else 2)
                if model == "gemma":
                    self.assertIsNone(summary["pairs"][0]["median_reduction_pct"])

    def test_refuses_profiles_even_if_raw_lacks_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp), "gemma", "nsys")
            with self.assertRaisesRegex(ValueError, "profiled timings"):
                compare.summarize(manifest)

    def test_refuses_raw_profiled_and_mismatched_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp))
            raw_path = Path(tmp) / "run01/cg_on_pdl_on/raw.json"
            raw = json.loads(raw_path.read_text())
            raw["horizon_len"] = 128
            raw_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "differ"):
                compare.summarize(manifest)
            raw["profiled"] = True
            raw_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "profiled"):
                compare.summarize(manifest)

    def test_moved_result_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(Path(tmp) / "old")
            moved = Path(tmp) / "moved"
            manifest.parent.rename(moved)
            self.assertEqual(compare.summarize(moved / "manifest.json")["marginal_pdl_reduction_pct"], 20)


class ExecutionTests(unittest.TestCase):
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = runner.make_plan(runner.parse_args(["timesfm", "benchmark", "--out-dir", tmp]), {})
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.preflight(plan)

    def test_mocked_capture_and_benchmark(self):
        for mode in ("benchmark", "xprof", "nsys"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                plan = runner.make_plan(runner.parse_args(["timesfm", mode, "--out-dir", str(Path(tmp) / "new")]), {})

                def fake_run(command, env, log):
                    directory = log.parent
                    runner.write_json(directory / "raw.json", {"samples_ms": [9, 10], "profiled": mode != "benchmark"})
                    log.write_text("mocked process; no GPU work\n")
                    if mode == "xprof":
                        (directory / "test.xplane.pb").touch()
                    if mode == "nsys":
                        (directory / "trace.nsys-rep").touch()
                    return 0

                with mock.patch.object(runner, "preflight"), mock.patch.object(runner, "record_setup"), \
                     mock.patch.object(runner, "run_logged", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                    runner.execute(plan, {})
                saved = json.loads((Path(plan["out_dir"]) / "manifest.json").read_text())
                self.assertTrue(all(j["status"] == "complete" for j in saved["jobs"]))
                self.assertEqual((Path(plan["out_dir"]) / "summary.json").exists(), mode == "benchmark")

    def test_failed_process_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = runner.make_plan(runner.parse_args(["timesfm", "benchmark", "--out-dir", str(Path(tmp) / "new")]), {})
            with mock.patch.object(runner, "preflight"), mock.patch.object(runner, "record_setup"), \
                 mock.patch.object(runner, "run_logged", return_value=17), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    runner.execute(plan, {})
            saved = json.loads((Path(plan["out_dir"]) / "manifest.json").read_text())
            self.assertEqual(saved["jobs"][0]["status"], "failed")
            self.assertEqual(saved["jobs"][1]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
