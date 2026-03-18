"""
Custom Agent: create-submission-agent-em

Iterates over Python files in <code_folder> (one file = one step), sorted
alphabetically. Runs at most max_steps scripts; stops when all files are done.

Each script must accept:
    --train-dataset-path        path to train CSV
    --test-dataset-path         path to competition test CSV
    --output-submission-path    where to write the submission CSV
    (+ any additional_args)

After each script the agent validates and grades the submission, then reports
the MAE score to Emily via send_iteration_result.

agent_config fields:
    competition_id       (str)   Competition folder name.
                                 Default: "ventilator-pressure-prediction"
    code_folder_path     (str)   Path to the folder containing scripts to run.
                                 Absolute, or relative to the agent script.
                                 Default: <workspace>/codes
    train_dataset_path   (str)   Path to train CSV.
                                 Default: <workspace>/train.csv
    additional_args      (list)  Extra CLI args appended to every invocation.
                                 Default: []
"""

import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
    """
    Return the project root:
      - /  when running inside Emily (where /workspace exists)
      - the directory containing this script when running locally
    """
    if Path("/workspace").is_dir():
        return Path("/")
    return Path(__file__).parent


def _get_workspace_dir() -> Path:
    return _get_base_dir() / "workspace"


def _get_competition_dir(competition_id: str) -> Path:
    return _get_base_dir() / competition_id


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_archives(code_folder: Path) -> None:
    """
    Extract any zip / tar / tar.gz / tgz / tar.bz2 / gz archives found
    directly in code_folder, then flatten all resulting .py files to the
    root of code_folder (overwrite on name collision).
    Archives and temporary sub-directories are removed after extraction.
    """
    import gzip
    import shutil
    import tarfile
    import zipfile

    ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2"}

    archives = [
        f for f in code_folder.iterdir()
        if f.is_file() and f.suffix.lower() in ARCHIVE_SUFFIXES
    ]

    if not archives:
        return

    for archive in archives:
        tmp_dir = code_folder / f"_tmp_{archive.stem}"
        tmp_dir.mkdir()
        name = archive.name.lower()

        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(tmp_dir)

            elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
                with tarfile.open(archive) as tf:
                    tf.extractall(tmp_dir)

            elif name.endswith(".gz"):
                # Single-file gzip (e.g. script.py.gz)
                out_name = archive.stem  # strips the .gz
                with gzip.open(archive, "rb") as gz_in:
                    (tmp_dir / out_name).write_bytes(gz_in.read())

            print(f"  [extract] {archive.name} → {tmp_dir.name}/")

            # Flatten: move every .py file from the extracted tree to code_folder
            for py_file in tmp_dir.rglob("*.py"):
                dest = code_folder / py_file.name
                shutil.move(str(py_file), str(dest))
                print(f"    ← {py_file.relative_to(tmp_dir)}  →  {dest.name}")

            # Clean up macOS AppleDouble metadata files (._foo.py) injected by
            # Finder/Archive Utility. If the real foo.py also exists → delete ._foo.py.
            # If only ._foo.py exists (edge case) → rename it to foo.py.
            for dot_file in list(code_folder.glob("._*.py")):
                real_name = dot_file.name[2:]  # strip leading ._
                real_file = code_folder / real_name
                if real_file.exists():
                    dot_file.unlink()
                    print(f"    [cleanup] removed macOS metadata file: {dot_file.name}")
                else:
                    dot_file.rename(real_file)
                    print(f"    [cleanup] renamed {dot_file.name} → {real_name}")

        except Exception as e:
            print(f"  [extract] WARNING: could not extract {archive.name}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            archive.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CUDA memory
# ---------------------------------------------------------------------------

def _clear_cuda_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("  [cuda] memory cleared")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Submission grading
# ---------------------------------------------------------------------------

def _validate_and_grade(
    submission_path: Path,
    sample_submission_path: Path,
    private_test_path: Path,
    grade_py_path: Path,
) -> Tuple[bool, Optional[float], str]:
    """
    Returns (format_valid, score_or_None, human_readable_message).
    score is None when format is invalid or grading failed.
    """
    # --- 1. Load files ---
    try:
        submission = pd.read_csv(submission_path)
    except Exception as e:
        return False, None, f"Cannot read submission file: {e}"

    try:
        sample = pd.read_csv(sample_submission_path)
    except Exception as e:
        return False, None, f"Cannot read sample submission: {e}"

    # --- 2. Format checks ---
    if set(submission.columns) != set(sample.columns):
        return (
            False,
            None,
            f"Column mismatch — expected {sorted(sample.columns.tolist())}, "
            f"got {sorted(submission.columns.tolist())}",
        )

    if len(submission) != len(sample):
        return (
            False,
            None,
            f"Row count mismatch — expected {len(sample)}, got {len(submission)}",
        )

    # --- 3. Grade ---
    if not private_test_path.exists():
        return True, None, "Valid format but private_test.csv not found — skipping grade"

    if not grade_py_path.exists():
        return True, None, "Valid format but grade.py not found — skipping grade"

    try:
        private_test = pd.read_csv(private_test_path)

        spec = importlib.util.spec_from_file_location("_competition_grade", grade_py_path)
        grade_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grade_module)

        score = grade_module.grade(submission, private_test)
        return True, float(score), f"MAE = {score:.6f}"
    except Exception as e:
        return True, None, f"Valid format but grading raised an error: {e}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CreateSubmissionAgentEmAgent(BaseAgent):
    """
    For each Python file in <code_folder> (alphabetical, one file per step):
      1. Runs:  python <file>
                  --train-dataset-path      <train>
                  --test-dataset-path       <test>
                  --output-submission-path  <workspace>/submissions/submission_<stem>.csv
                  [additional_args]
      2. Reads the submission from the explicit output path it just passed
      3. Validates & grades submission
      4. Reports score to Emily via send_iteration_result / send_experiment_completed

    Stops after min(max_steps, len(scripts)) steps — never loops beyond available files.
    Scripts must honour --output-submission-path so they can also be run standalone.
    """

    async def start(self):
        competition_id: str = self.agent_config.get(
            "competition_id", "ventilator-pressure-prediction"
        )
        additional_args: List[str] = self.agent_config.get("additional_args", [])

        workspace_dir = _get_workspace_dir()
        competition_dir = _get_competition_dir(competition_id)

        # Code folder: explicit config, or default to workspace/codes
        raw_code_folder = self.agent_config.get("code_folder_path")
        if raw_code_folder:
            code_folder = Path(raw_code_folder)
            if not code_folder.is_absolute():
                # Relative paths are anchored to the project root (same as workspace)
                code_folder = _get_base_dir() / code_folder
        else:
            code_folder = workspace_dir / "codes"

        # Paths inside competition folder
        test_dataset_path = competition_dir / "test.csv"
        sample_submission_path = competition_dir / "sample_submission.csv"
        private_test_path = competition_dir / "private_test.csv"
        grade_py_path = competition_dir / "grade.py"

        # train dataset: explicit config or default to workspace/train.csv
        raw_train_path = self.agent_config.get("train_dataset_path")
        if raw_train_path:
            train_dataset_path = Path(raw_train_path)
            if not train_dataset_path.is_absolute():
                train_dataset_path = _get_base_dir() / train_dataset_path
        else:
            train_dataset_path = workspace_dir / "train.csv"

        def _make_run_dir(base_name: str) -> Path:
            candidate = workspace_dir / base_name
            if candidate.exists():
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                candidate = workspace_dir / f"{base_name}_{ts}"
            candidate.mkdir(parents=True)
            return candidate

        submissions_dir = _make_run_dir(f"submissions_{self.experiment_id}")
        grid_dir = _make_run_dir(f"grid_{self.experiment_id}")

        # Extract any archives in code folder before iterating
        _extract_archives(code_folder)

        # Collect Python files in code folder (alphabetical = deterministic order)
        python_files: List[Path] = sorted(code_folder.glob("*.py"))

        # Run at most max_steps scripts; stop when files are exhausted
        effective_steps = min(self.max_steps, len(python_files))

        print(f"Workspace:        {workspace_dir}")
        print(f"Code folder:      {code_folder}  ({len(python_files)} script(s))")
        print(f"Submissions dir:  {submissions_dir}")
        print(f"Competition dir:  {competition_dir}")
        print(f"Train dataset:    {train_dataset_path}")
        print(f"Test dataset:     {test_dataset_path}")
        print(f"Python files:     {len(python_files)}")
        print(f"Effective steps:  {effective_steps}")

        await self.send_initial_messages(
            system_message=(
                "You are an ML submission runner. "
                "Each step executes one Python training script and grades its submission."
            ),
            user_message=(
                f"Competition: {competition_id}\n"
                f"Running {effective_steps}/{len(python_files)} script(s) from {code_folder.name}.\n"
                f"Train data: {train_dataset_path}\n"
                f"Test data:  {test_dataset_path}"
            ),
            step_number=0,
        )

        best_score: Optional[float] = None
        best_script: Optional[str] = None

        for step, py_file in enumerate(python_files[:effective_steps], 1):
            if self.is_aborted:
                break

            print(f"\n{'='*60}")
            print(f"Step {step}/{effective_steps}: {py_file.name}")
            print(f"{'='*60}")

            # Clear GPU memory before each step
            _clear_cuda_memory()

            # Output path that the script must write to
            submission_path = submissions_dir / f"submission_{py_file.stem}.csv"

            # Build command
            cmd = [
                sys.executable,
                str(py_file),
                "--train-dataset-path", str(train_dataset_path),
                "--test-dataset-path", str(test_dataset_path),
                "--output-submission-path", str(submission_path),
            ] + [str(a) for a in additional_args]

            action_id = f"action_{self.experiment_id}_{step}"
            thought = (
                f"Running {py_file.name} with train={train_dataset_path.name}, "
                f"test={test_dataset_path.name}"
            )

            await self.send_action_received(
                step_number=step,
                action_id=action_id,
                action_type="run_script",
                action_message={
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": None,
                    "completion_details": None,
                },
            )

            # --- Execute the script ---
            observation_lines = [f"$ {' '.join(cmd)}\n"]
            score: Optional[float] = None
            error_msg: Optional[str] = None
            elapsed_seconds: Optional[float] = None

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hours per script
                )
                elapsed_seconds = round(time.monotonic() - t0, 2)
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if stdout:
                    observation_lines.append(f"[stdout]\n{stdout}")
                if stderr:
                    observation_lines.append(f"[stderr]\n{stderr}")
                observation_lines.append(f"\nExit code: {proc.returncode}  ({elapsed_seconds}s)")

                if proc.returncode != 0:
                    error_msg = f"Script exited with code {proc.returncode}"

            except subprocess.TimeoutExpired:
                elapsed_seconds = round(time.monotonic() - t0, 2)
                observation_lines.append("TIMEOUT: script exceeded 2-hour limit")
                error_msg = "Script timed out"
            except Exception as e:
                elapsed_seconds = round(time.monotonic() - t0, 2)
                observation_lines.append(f"ERROR launching script: {e}")
                error_msg = str(e)

            # --- Find submission file (path was passed explicitly to the script) ---
            format_valid = False
            grade_msg = "submission file not found"

            if submission_path.exists():
                format_valid, score, grade_msg = _validate_and_grade(
                    submission_path=submission_path,
                    sample_submission_path=sample_submission_path,
                    private_test_path=private_test_path,
                    grade_py_path=grade_py_path,
                )
                observation_lines.append(f"\n[grading] {submission_path.name}: {grade_msg}")
                if not format_valid:
                    observation_lines.append("  → Submission format invalid, skipping grade")
            else:
                observation_lines.append(f"\n[grading] {submission_path.name} not found — skipping")

            observation = "\n".join(observation_lines)
            print(observation)

            # --- Write grid JSON ---
            grid_entry = {
                "python_file": str(py_file),
                "submission_file": str(submission_path) if submission_path.exists() else None,
                "score": score,
                "format_valid": format_valid,
                "grade_message": grade_msg,
                "execution_time_seconds": elapsed_seconds,
                "error": error_msg,
            }
            grid_path = grid_dir / f"metric_{py_file.stem}.json"
            grid_path.write_text(json.dumps(grid_entry, indent=2))
            print(f"  [grid] {grid_path.name} written")

            # Track best (lower MAE is better)
            if score is not None:
                if best_score is None or score < best_score:
                    best_score = score
                    best_script = py_file.name

            await self.send_step_finished(
                step_number=step,
                action_id=action_id,
                action_type="run_script",
                observation_content=observation,
                tool_call_id=f"call_{step}",
                error=error_msg,
            )

            # Report iteration result (experiment stays RUNNING)
            await self.send_iteration_result(
                success=(error_msg is None and format_valid),
                summary=(
                    f"{py_file.name}: {grade_msg}"
                    if submission_path.exists()
                    else f"{py_file.name}: no submission produced"
                ),
                score=score,
                approach=f"Script: {py_file.name}",
                step_number=step,
            )

            self.current_step = step

        # Final result
        await self.send_experiment_completed(
            success=best_score is not None,
            summary=(
                f"Best score: {best_score:.6f} (MAE) from {best_script}"
                if best_score is not None
                else "No valid graded submission produced"
            ),
            score=best_score,
            approach=f"Ran {effective_steps} script(s) from workspace",
            step_number=self.current_step,
        )

        print(f"\nDone. Best MAE: {best_score} ({best_script})")

    async def continue_agent(
        self,
        user_message: str,
        new_max_steps: int,
        step_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ):
        print(f"Received user message: {user_message}")
        self.max_steps = new_max_steps

        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1

        await self.send_step_finished(
            step_number=self.current_step,
            user_message=user_message,
            git_branch=branch_name,
        )

        # Re-run the remaining scripts with the new max_steps budget
        await self.start()

    async def abort(self):
        print("Aborting create-submission-agent-em...")
        self.is_aborted = True
        await self.send_experiment_aborted(
            reason="Aborted by user",
            last_step=self.current_step,
        )


def create_agent(
    experiment_id: str,
    project_id: str,
    problem_statement: str,
    max_steps: int,
    api_keys: Dict[str, str],
    webhook_url: Optional[str] = None,
    agent_config: Dict[str, Any] = None,
    jwt_token: str = None,
) -> BaseAgent:
    return CreateSubmissionAgentEmAgent(
        experiment_id=experiment_id,
        project_id=project_id,
        problem_statement=problem_statement,
        max_steps=max_steps,
        api_keys=api_keys,
        webhook_url=webhook_url,
        agent_config=agent_config,
        jwt_token=jwt_token,
    )
