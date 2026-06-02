"""Concrete evolving governor implementation for KernelBench-compatible runs."""

        recorder.start()
        try:
            def _do_iteration(attempt: int) -> None:
                holder.set_iteration(attempt)
                l1_text = read_l1(l1_path)
                l0_text = self._format_l0_for_coder_prompt(l0)
                iteration_context = self._build_iteration_context(
                    attempt=attempt,
                    best_speedup=best_speedup,
                    best_correct=best_correct,
                    best_compiled=best_compiled,
                )
                latest_eval_feedback = self._latest_eval_feedback(records)
                selected_l1_entries: list[dict[str, str]] | None = None

                l1_entries = read_l1_jsonl(l1_path)
                if self.config.enable_l1_extractor and l1_entries:
                    max_entries = max(1, int(self.config.extractor_max_memories))
                    fallback_selected = l1_entries[-max_entries:]
                    extractor_messages = self._build_extractor_messages(
                        task_prompt=task_prompt,
                        l1_entries=l1_entries,
                        iteration_context=iteration_context,
                        latest_eval_feedback=latest_eval_feedback,
                    )
                    try:
                        extractor_model_id = (
                            resolve_nvidia_model_id(self.config.extractor_model)
                            if self.config.extractor_model
                            else None
                        )
                        extractor_raw, _extractor_tokens, extractor_meta = call_extractor_with_meta(
                            extractor_messages,
                            max_tokens=self.config.extractor_max_tokens,
                            timeout_sec=self.config.extractor_timeout_sec,
                            model_id=extractor_model_id,
                        )
                        recorder.record_llm_turn(
                            iteration=attempt,
                            phase="extractor",
                            messages=extractor_messages,
                            assistant_text=extractor_raw,
                            extra=extractor_meta,
                        )
                        selected_ids = self._parse_selected_entry_ids(
                            extractor_raw,
                            valid_ids={entry.get("entry_id", "") for entry in l1_entries},
                        )
                        if selected_ids:
                            by_id = {
                                str(entry.get("entry_id", "")): entry
                                for entry in l1_entries
                                if entry.get("entry_id")
                            }
                            selected_l1_entries = [
                                by_id[entry_id]
                                for entry_id in selected_ids
                                if entry_id in by_id
                            ][:max_entries]
                    except Exception as exc:
                        append_l0(
                            l0,
                            "terminal",
                            f"extractor_selection_error: {type(exc).__name__}: {exc}",
                        )

                    if not selected_l1_entries:
                        selected_l1_entries = fallback_selected

                l1_for_prompt = l1_text
                if selected_l1_entries:
                    l1_for_prompt = "(Extractor-selected L1 entries are provided in the section below.)"

                coder_prompt = self._get_coder_prompt(
                    task_prompt,
                    l1_for_prompt,
                    l0_text,
                    selected_l1_entries=selected_l1_entries,
                    allowed_actions=ALLOWED_CODER_ACTIONS,
                    iteration_context=iteration_context,
                    latest_eval_feedback=latest_eval_feedback,
                )
                coder_messages = [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": coder_prompt},
                ]

                try:
                    raw, _tokens, coder_meta = call_coder_with_meta(
                        coder_messages,
                        max_tokens=self.config.coder_max_tokens,
                        timeout_sec=self.config.coder_timeout_sec,
                    )
                except Exception as exc:
                    nonlocal fatal_error_count, run_error
                    fatal_error_count += 1
                    err = f"coder_call_error: {type(exc).__name__}: {exc}"
                    append_l0(l0, "terminal", err)
                    metrics_iteration = {
                        "attempt": attempt,
                        "compiled": False,
                        "correct": False,
                        "speedup": 0.0,
                        "error": err,
                    }
                    holder.update_iteration_metrics(metrics_iteration)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="coder_call",
                        terminal_output=err,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
                    recorder.record_iteration_snapshot(
                        iteration=attempt,
                        l0_entries=l0_entries_to_json_serializable(l0),
                        l1_text=l1_text,
                        l1_path=str(l1_path),
                        metrics_iteration=metrics_iteration,
                        metrics_best=holder.get_snapshot().get("metrics_best", {}),
                    )
                    if self._handle_fatal_error(exc) or fatal_error_count >= self.config.max_fatal_errors:
                        run_error = err
                        raise
                    return

                recorder.record_llm_turn(
                    iteration=attempt,
                    phase="coder",
                    messages=coder_messages,
                    assistant_text=raw,
                    extra=coder_meta,
                )

                diagnosis_text = self._extract_optional_tag(raw, "diagnosis")
                if diagnosis_text:
                    append_l0(l0, "system", f"coder_diagnosis={diagnosis_text}")

                hypothesis_text = self._extract_optional_tag(raw, "hypothesis")
                if hypothesis_text:
                    append_l0(l0, "system", f"coder_hypothesis={hypothesis_text}")

                action_text = self._extract_optional_tag(raw, "action")
                if not action_text:
                    action_text = self._fallback_action(attempt=attempt, records=records)

                action_norm = action_text.strip().lower()
                if action_norm not in ALLOWED_CODER_ACTIONS:
                    action_norm = self._fallback_action(attempt=attempt, records=records)
                append_l0(l0, "system", f"coder_action={action_norm}")

                reasoning_text = self._extract_optional_tag(raw, "reasoning")
                if not reasoning_text:
                    reasoning_text = self._fallback_reasoning(raw, action_norm)
                append_l0(l0, "system", f"coder_reasoning={reasoning_text}")

                code, extract_err = normalize_extracted_python(raw)
                if extract_err or not code:
                    extraction_error = extract_err or "no code extracted"
                    extract_terminal = f"extract_error={extraction_error}; raw={(raw or '')[:1000]}"
                    append_l0(l0, "terminal", extract_terminal)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="extract",
                        terminal_output=extract_terminal,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
                    eval_result = KBEvalResult(
                        compiled=False,
                        correct=False,
                        error_message=extraction_error,
                    )
                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code="# extraction_failed",
                        evaluation=eval_result,
                    )
                    records.append(record)
                else:
                    append_l0(l0, "code", code)
                    eval_result = self._evaluate_candidate(code, attempt=attempt)
                    runtime_value = (
                        float(eval_result.runtime)
                        if eval_result.runtime is not None
                        else None
                    )
                    ref_runtime_value = (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    )
                    terminal_log = (
                        f"KERNEL_BENCH_CORRECT: {eval_result.correct}\n"
                        f"KERNEL_BENCH_SPEEDUP: {float(eval_result.speedup or 0.0):.6f}\n"
                        f"KERNEL_BENCH_RUNTIME: {runtime_value if runtime_value is not None else 'n/a'}\n"
                        f"KERNEL_BENCH_REF_RUNTIME: {ref_runtime_value if ref_runtime_value is not None else 'n/a'}\n"
                    )
                    if eval_result.error_message:
                        terminal_log += f"KERNEL_BENCH_ERROR: {eval_result.error_message}\n"
                    if eval_result.terminal_output:
                        terminal_log += (
                            "KERNEL_BENCH_EVAL_TERMINAL_OUTPUT:\n"
                            f"{eval_result.terminal_output}\n"
                        )
                    append_l0(l0, "terminal", terminal_log)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="evaluation",
                        terminal_output=eval_result.terminal_output or terminal_log,
                        extra={
                            "compiled": bool(eval_result.compiled),
                            "correct": bool(eval_result.correct),
                            "speedup": float(eval_result.speedup or 0.0),
                            "runtime": runtime_value,
                            "ref_runtime": ref_runtime_value,
                            "error": eval_result.error_message,
                        },
                    )

                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code=code,
                        evaluation=eval_result,
                    )
                    records.append(record)

                    speedup = float(eval_result.speedup or 0.0)
                    if eval_result.correct and speedup >= best_speedup:
                        best_speedup = speedup
                        best_correct = bool(eval_result.correct)
                        best_compiled = bool(eval_result.compiled)
                        best_code = code
                        best_runtime = float(eval_result.runtime) if eval_result.runtime is not None else -1.0
                        best_runtime_stats = dict(eval_result.runtime_stats)
                        best_metadata = dict(eval_result.metadata)
                        best_code_path = workspace_dir / f"best_iter_{attempt}.py"
                        best_code_path.write_text(code, encoding="utf-8")

                metrics_iteration = {
                    "attempt": attempt,
                    "compiled": bool(eval_result.compiled),
                    "correct": bool(eval_result.correct),
                    "speedup": float(eval_result.speedup or 0.0),
                    "runtime": float(eval_result.runtime) if eval_result.runtime is not None else None,
                    "ref_runtime": (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    ),
                    "error": eval_result.error_message,
                }
                metrics_best = {
                    "compiled": best_compiled,
                    "correct": best_correct,
                    "speedup": best_speedup,
                    "runtime": best_runtime if best_runtime >= 0 else None,
                }
                holder.update_iteration_metrics(metrics_iteration)
                holder.update_best(metrics_best)

                self._last_promoted_count = maybe_promote_l0_to_l1(
                    l0,
                    l1_path=l1_path,
                    entry_threshold=self.config.promote_entry_threshold,
                    token_budget=self.config.promote_token_budget,
                    summarizer_system_prompt=SUMMARIZER_SYSTEM_PROMPT,
                    build_summarizer_user_message=build_summarizer_user_message,
                    summarizer_max_tokens=self.config.summarizer_max_tokens,
                    summarizer_timeout_sec=self.config.summarizer_timeout_sec,
                    enable_promotion=self.config.enable_promotion,
                    catch_summarizer_errors=True,
                    verbose=self.config.verbose,
                    log_prefix="[kb-governor]",
                    source=(
                        f"Level {self.config.level} problem {self.config.problem_id}"
                    ),
                    on_summarizer_round=recorder.summarizer_callback(attempt),
                    on_l0_cleared_without_l1=recorder.flush_without_l1_callback(attempt),
                    clear_l0_after_promotion=False,
                    last_promoted_count=self._last_promoted_count,
                )

                recorder.record_iteration_snapshot(
                    iteration=attempt,
                    l0_entries=l0_entries_to_json_serializable(l0),
                    l1_text=read_l1(l1_path),
                    l1_path=str(l1_path),
                    metrics_iteration=metrics_iteration,
                    metrics_best=metrics_best,
                )

                if self.config.verbose:
                    print(
                        f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f} "
                        f"runtime={float(eval_result.runtime):.6f}"
                        if eval_result.runtime is not None
                        else f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f} runtime=n/a"
                    )

            # use the generic run_loop helper from BaseEvolvingGovernor
            self.run_loop(_do_iteration)
        except Exception:
            run_error = traceback.format_exc()
        finally:

        error_message = None
        if not correct:
            error_message = (
                metadata.get("compilation_error")
                or metadata.get("runtime_error")
                or metadata.get("correctness_issue")
            )
            if error_message is not None:
                error_message = str(error_message)

        runtime_value: float | None
        ref_runtime_value: float | None
        try:
            runtime_value = float(runtime) if runtime is not None and float(runtime) >= 0 else None
        except Exception:
            runtime_value = None

        try:
            ref_runtime_value = (
                float(ref_runtime)
                if ref_runtime is not None and float(ref_runtime) >= 0
                else None
            )
        except Exception:
            ref_runtime_value = None

        normalized_metadata = dict(metadata)
        if terminal_output:
            normalized_metadata.setdefault("evaluation_terminal_output", terminal_output)

        return KBEvalResult(
            compiled=compiled,
            correct=correct,
            speedup=speedup,
            error_message=error_message,
            runtime=runtime_value,
            ref_runtime=ref_runtime_value,
            terminal_output=terminal_output,
            runtime_stats=dict(runtime_stats),
            metadata=normalized_metadata,
        )

    def _evaluate_candidate(self, code: str, *, attempt: int) -> KBEvalResult:
        normalized = code.strip()
        if not normalized:
            return KBEvalResult(
                compiled=False,
                correct=False,
                error_message="empty candidate code",
            )

        if not self.config.reference_code:
            return KBEvalResult(
                compiled=False,
                correct=False,
                error_message="missing reference_code for KernelBench evaluation",
            )

        build_dir = (
            Path(self.config.results_root)
            / self.config.run_name
            / "builds"
            / f"l{self.config.level}_p{self.config.problem_id}_iter{attempt}_{uuid4().hex[:8]}"
        )
        build_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "reference_code": self.config.reference_code,
            "candidate_code": normalized,
            "backend": self.config.backend,
            "precision": self.config.precision,
            "build_dir": str(build_dir),
        }

        self.reserver.release()
        try:
            if self.config.isolate_evaluation_process:
                eval_payload = self._evaluate_in_subprocess(payload)
            else:
                eval_payload = _run_kernelbench_eval(payload)
        finally:
            self.reserver.acquire()

        return self._build_eval_result(eval_payload)

    def _get_summarizer_prompt(self, result: KBEvalResult) -> str:
        terminal = (
            f"compiled={result.compiled}, correct={result.correct}, "
            f"speedup={result.speedup if result.speedup is not None else 'n/a'}, "
            f"error={result.error_message or 'none'}"
        )
        return build_summarizer_user_message(terminal)

    def _handle_fatal_error(self, error: Exception) -> bool:
        return isinstance(error, (MemoryError, SystemError))

    def run(self, *, task_prompt: str) -> KBGovernorResult:
        run_dir = Path(self.config.results_root) / self.config.run_name
        workspace_dir = (
            run_dir
            / "workspaces"
            / f"level_{self.config.level}_problem_{self.config.problem_id}"
        )
        workspace_dir.mkdir(parents=True, exist_ok=True)

        l1_path = self.config.shared_l1_path or run_dir / "shared_l1.txt"
        l1_path.parent.mkdir(parents=True, exist_ok=True)
        if not l1_path.exists():
            l1_path.write_text("# Shared L1 journal for evolving KernelBench batch\n", encoding="utf-8")

        holder = BestMetricsHolder()
        recorder = BenchmarkRunRecorder(
            RunRecorderConfig(
                output_dir=workspace_dir,
                time_sample_interval_sec=self.config.run_recorder_time_sample_interval_sec,
            ),
            holder,
            run_metadata={
                "benchmark": "kernelbench",
                "level": self.config.level,
                "problem_id": self.config.problem_id,
                "backend": self.config.backend,
                "precision": self.config.precision,
            },
        )

        l0 = fresh_l0_for_problem()
        records: list[KBIterationRecord] = []
        best_speedup = 0.0
        best_correct = False
        best_compiled = False
        best_code: str | None = None
        best_code_path: Path | None = None
        best_runtime = -1.0
        best_runtime_stats: dict[str, Any] = {}
        best_metadata: dict[str, Any] = {}
        run_error: str | None = None
        fatal_error_count = 0

        recorder.start()
        try:
            for attempt in range(1, self.config.max_iterations + 1):
                holder.set_iteration(attempt)
                l1_text = read_l1(l1_path)
                l0_text = self._format_l0_for_coder_prompt(l0)
                iteration_context = self._build_iteration_context(
                    attempt=attempt,
                    best_speedup=best_speedup,
                    best_correct=best_correct,
                    best_compiled=best_compiled,
                )
                latest_eval_feedback = self._latest_eval_feedback(records)
                selected_l1_entries: list[dict[str, str]] | None = None

                l1_entries = read_l1_jsonl(l1_path)
                if self.config.enable_l1_extractor and l1_entries:
                    max_entries = max(1, int(self.config.extractor_max_memories))
                    fallback_selected = l1_entries[-max_entries:]
                    extractor_messages = self._build_extractor_messages(
                        task_prompt=task_prompt,
                        l1_entries=l1_entries,
                        iteration_context=iteration_context,
                        latest_eval_feedback=latest_eval_feedback,
                    )
                    try:
                        extractor_model_id = (
                            resolve_nvidia_model_id(self.config.extractor_model)
                            if self.config.extractor_model
                            else None
                        )
                        extractor_raw, _extractor_tokens, extractor_meta = call_extractor_with_meta(
                            extractor_messages,
                            max_tokens=self.config.extractor_max_tokens,
                            timeout_sec=self.config.extractor_timeout_sec,
                            model_id=extractor_model_id,
                        )
                        recorder.record_llm_turn(
                            iteration=attempt,
                            phase="extractor",
                            messages=extractor_messages,
                            assistant_text=extractor_raw,
                            extra=extractor_meta,
                        )
                        selected_ids = self._parse_selected_entry_ids(
                            extractor_raw,
                            valid_ids={entry.get("entry_id", "") for entry in l1_entries},
                        )
                        if selected_ids:
                            by_id = {
                                str(entry.get("entry_id", "")): entry
                                for entry in l1_entries
                                if entry.get("entry_id")
                            }
                            selected_l1_entries = [
                                by_id[entry_id]
                                for entry_id in selected_ids
                                if entry_id in by_id
                            ][:max_entries]
                    except Exception as exc:
                        append_l0(
                            l0,
                            "terminal",
                            f"extractor_selection_error: {type(exc).__name__}: {exc}",
                        )

                    if not selected_l1_entries:
                        selected_l1_entries = fallback_selected

                l1_for_prompt = l1_text
                if selected_l1_entries:
                    l1_for_prompt = "(Extractor-selected L1 entries are provided in the section below.)"

                coder_prompt = self._get_coder_prompt(
                    task_prompt,
                    l1_for_prompt,
                    l0_text,
                    selected_l1_entries=selected_l1_entries,
                    allowed_actions=ALLOWED_CODER_ACTIONS,
                    iteration_context=iteration_context,
                    latest_eval_feedback=latest_eval_feedback,
                )
                coder_messages = [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": coder_prompt},
                ]

                try:
                    raw, _tokens, coder_meta = call_coder_with_meta(
                        coder_messages,
                        max_tokens=self.config.coder_max_tokens,
                        timeout_sec=self.config.coder_timeout_sec,
                    )
                except Exception as exc:
                    fatal_error_count += 1
                    err = f"coder_call_error: {type(exc).__name__}: {exc}"
                    append_l0(l0, "terminal", err)
                    metrics_iteration = {
                        "attempt": attempt,
                        "compiled": False,
                        "correct": False,
                        "speedup": 0.0,
                        "error": err,
                    }
                    holder.update_iteration_metrics(metrics_iteration)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="coder_call",
                        terminal_output=err,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
                    recorder.record_iteration_snapshot(
                        iteration=attempt,
                        l0_entries=l0_entries_to_json_serializable(l0),
                        l1_text=l1_text,
                        l1_path=str(l1_path),
                        metrics_iteration=metrics_iteration,
                        metrics_best=holder.get_snapshot().get("metrics_best", {}),
                    )
                    if self._handle_fatal_error(exc) or fatal_error_count >= self.config.max_fatal_errors:
                        run_error = err
                        break
                    continue

                recorder.record_llm_turn(
                    iteration=attempt,
                    phase="coder",
                    messages=coder_messages,
                    assistant_text=raw,
                    extra=coder_meta,
                )

                diagnosis_text = self._extract_optional_tag(raw, "diagnosis")
                if diagnosis_text:
                    append_l0(l0, "system", f"coder_diagnosis={diagnosis_text}")

                hypothesis_text = self._extract_optional_tag(raw, "hypothesis")
                if hypothesis_text:
                    append_l0(l0, "system", f"coder_hypothesis={hypothesis_text}")

                action_text = self._extract_optional_tag(raw, "action")
                if not action_text:
                    action_text = self._fallback_action(attempt=attempt, records=records)

                action_norm = action_text.strip().lower()
                if action_norm not in ALLOWED_CODER_ACTIONS:
                    action_norm = self._fallback_action(attempt=attempt, records=records)
                append_l0(l0, "system", f"coder_action={action_norm}")

                reasoning_text = self._extract_optional_tag(raw, "reasoning")
                if not reasoning_text:
                    reasoning_text = self._fallback_reasoning(raw, action_norm)
                append_l0(l0, "system", f"coder_reasoning={reasoning_text}")

                code, extract_err = normalize_extracted_python(raw)
                if extract_err or not code:
                    extraction_error = extract_err or "no code extracted"
                    extract_terminal = f"extract_error={extraction_error}; raw={(raw or '')[:1000]}"
                    append_l0(l0, "terminal", extract_terminal)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="extract",
                        terminal_output=extract_terminal,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
                    eval_result = KBEvalResult(
                        compiled=False,
                        correct=False,
                        error_message=extraction_error,
                    )
                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code="# extraction_failed",
                        evaluation=eval_result,
                    )
                    records.append(record)
                else:
                    append_l0(l0, "code", code)
                    eval_result = self._evaluate_candidate(code, attempt=attempt)
                    runtime_value = (
                        float(eval_result.runtime)
                        if eval_result.runtime is not None
                        else None
                    )
                    ref_runtime_value = (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    )
                    terminal_log = (
                        f"KERNEL_BENCH_CORRECT: {eval_result.correct}\n"
                        f"KERNEL_BENCH_SPEEDUP: {float(eval_result.speedup or 0.0):.6f}\n"
                        f"KERNEL_BENCH_RUNTIME: {runtime_value if runtime_value is not None else 'n/a'}\n"
                        f"KERNEL_BENCH_REF_RUNTIME: {ref_runtime_value if ref_runtime_value is not None else 'n/a'}\n"
                    )
                    if eval_result.error_message:
                        terminal_log += f"KERNEL_BENCH_ERROR: {eval_result.error_message}\n"
                    if eval_result.terminal_output:
                        terminal_log += (
                            "KERNEL_BENCH_EVAL_TERMINAL_OUTPUT:\n"
                            f"{eval_result.terminal_output}\n"
                        )
                    append_l0(l0, "terminal", terminal_log)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="evaluation",
                        terminal_output=eval_result.terminal_output or terminal_log,
                        extra={
                            "compiled": bool(eval_result.compiled),
                            "correct": bool(eval_result.correct),
                            "speedup": float(eval_result.speedup or 0.0),
                            "runtime": runtime_value,
                            "ref_runtime": ref_runtime_value,
                            "error": eval_result.error_message,
                        },
                    )

                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code=code,
                        evaluation=eval_result,
                    )
                    records.append(record)

                    speedup = float(eval_result.speedup or 0.0)
                    if eval_result.correct and speedup >= best_speedup:
                        best_speedup = speedup
                        best_correct = bool(eval_result.correct)
                        best_compiled = bool(eval_result.compiled)
                        best_code = code
                        best_runtime = float(eval_result.runtime) if eval_result.runtime is not None else -1.0
                        best_runtime_stats = dict(eval_result.runtime_stats)
                        best_metadata = dict(eval_result.metadata)
                        best_code_path = workspace_dir / f"best_iter_{attempt}.py"
                        best_code_path.write_text(code, encoding="utf-8")

                metrics_iteration = {
                    "attempt": attempt,
                    "compiled": bool(eval_result.compiled),
                    "correct": bool(eval_result.correct),
                    "speedup": float(eval_result.speedup or 0.0),
                    "runtime": float(eval_result.runtime) if eval_result.runtime is not None else None,
                    "ref_runtime": (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    ),
                    "error": eval_result.error_message,
                }
                metrics_best = {
                    "compiled": best_compiled,
                    "correct": best_correct,
                    "speedup": best_speedup,
                    "runtime": best_runtime if best_runtime >= 0 else None,
                }
                holder.update_iteration_metrics(metrics_iteration)
                holder.update_best(metrics_best)

                self._last_promoted_count = maybe_promote_l0_to_l1(
                    l0,
                    l1_path=l1_path,
                    entry_threshold=self.config.promote_entry_threshold,
                    token_budget=self.config.promote_token_budget,
                    summarizer_system_prompt=SUMMARIZER_SYSTEM_PROMPT,
                    build_summarizer_user_message=build_summarizer_user_message,
                    summarizer_max_tokens=self.config.summarizer_max_tokens,
                    summarizer_timeout_sec=self.config.summarizer_timeout_sec,
                    enable_promotion=self.config.enable_promotion,
                    catch_summarizer_errors=True,
                    verbose=self.config.verbose,
                    log_prefix="[kb-governor]",
                    source=(
                        f"Level {self.config.level} problem {self.config.problem_id}"
                    ),
                    on_summarizer_round=recorder.summarizer_callback(attempt),
                    on_l0_cleared_without_l1=recorder.flush_without_l1_callback(attempt),
                    clear_l0_after_promotion=False,
                    last_promoted_count=self._last_promoted_count,
                )

                recorder.record_iteration_snapshot(
                    iteration=attempt,
                    l0_entries=l0_entries_to_json_serializable(l0),
                    l1_text=read_l1(l1_path),
                    l1_path=str(l1_path),
                    metrics_iteration=metrics_iteration,
                    metrics_best=metrics_best,
                )

                if self.config.verbose:
                    print(
                        f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f} "
                        f"runtime={float(eval_result.runtime):.6f}"
                        if eval_result.runtime is not None
                        else f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f} runtime=n/a"
                    )

        except Exception:
            run_error = traceback.format_exc()
        finally:
            recorder.stop(
                final_metadata={
                    "best_speedup": best_speedup,
                    "best_correct": best_correct,
                    "best_compiled": best_compiled,
                    "error": run_error,
                }
            )

        return KBGovernorResult(
            level=self.config.level,
            problem_id=self.config.problem_id,
            backend=self.config.backend,
            precision=self.config.precision,
            best_speedup=best_speedup,
            best_correct=best_correct,
            best_compiled=best_compiled,
            best_code_path=str(best_code_path) if best_code_path else None,
            best_code=best_code,
            iterations_run=len(records),
            records=records,
            runtime=float(best_runtime),
            runtime_stats=best_runtime_stats,
            metadata=best_metadata,
            error=run_error,
        )


def governor_result_to_dict(result: KBGovernorResult) -> dict[str, Any]:
    return result.model_dump()


def safe_run_kb_governor(cfg: KBGovernorConfig, *, task_prompt: str) -> KBGovernorResult:
    try:
        return KBGovernor(cfg).run(task_prompt=task_prompt)
    except Exception:
        return KBGovernorResult(
            level=cfg.level,
            problem_id=cfg.problem_id,
            backend=cfg.backend,
            precision=cfg.precision,
            best_speedup=0.0,
            best_correct=False,
            best_compiled=False,
            best_code_path=None,
            best_code=None,
            iterations_run=0,
            records=[],
            runtime=-1.0,
            runtime_stats={},
            metadata={},
            error=traceback.format_exc(),
        )
