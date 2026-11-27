# Mission 08 — Cygnus X-1
## Implementation and Test Documentation

**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 8 — Core Agentic Design Patterns & State Memory Systems

## 1. Project overview

This project implements four native-Python modules for an agentic execution and virtual-memory architecture:

1. `rewoo_planner.py` — decoupled planning and dependency execution.
2. `reflexion_surgeon.py` — traceback parsing and test-grounded repair-loop support.
3. `memgpt_pager.py` — virtual context paging and tri-factor retrieval.
4. `sandbox_guardian.py` — AST-based static checks and memory provenance auditing.

The implementation uses Python's standard library and does not require external frameworks.

## 2. Part 1 — ReWOO planner

### Functions

- `parse_blueprint(blueprint_text)` reads `Plan:` descriptions and symbolic tool lines such as `#E1 = Search[query]`, returning `PlanStep` objects.
- `build_dependency_levels(steps)` finds `#E` references and groups independent steps into topological levels. It reports unknown references, duplicate variables, and cycles.
- `execute_dag(steps, tools)` substitutes resolved values and runs independent steps in a level concurrently. An optional callback can provide one replacement step after a runtime error.
- `compare_token_cost(k_steps, base_tokens, step_tokens, obs_tokens)` computes the assignment's ReAct and ReWOO formulas and returns token totals, percentage savings, and speedup ratio.

### Token formulas

ReAct:
`K * L0 + (K * (K - 1) / 2) * (Lstep + Lobs)`

ReWOO:
`2 * L0 + K * (Lstep + Lobs)`

The implementation uses integer arithmetic for the triangular sum.

## 3. Part 2 — Reflexion surgeon

### Functions

- `parse_traceback(stderr_text)` extracts the final file, line number, exception type, and fault summary where available.
- `generate_verbal_reflection(diag, failed_code)` creates a reflection based on the observed diagnostic and nearby source lines.
- `is_duplicate_patch(candidate_code, seen_hashes)` hashes whitespace-normalized source with SHA-256 and tracks repeated candidates when a shared set is supplied.
- `run_repair_loop(initial_code, tests, patcher, max_trials=5)` runs deterministic tests, keeps the latest three reflections, rejects repeated patches, and stops after the trial limit.

The test and patch functions are injected by the caller. This keeps the repair loop independent of any particular model or testing framework.

## 4. Part 3 — MemGPT virtual pager

`MemGPTPager` maintains core memory, a FIFO queue, an archival store, and interrupt messages.

- Core memory supports append and replace.
- Archival memory supports insert and ranked search.
- `check_memory_pressure()` warns when estimated token use reaches 70% of the budget and moves the oldest half of the FIFO queue into one consolidated archival node.
- `score_memory()` combines recency decay (`0.99 ** hours`), normalized importance (`importance / 10`), and cosine relevance.

The included token counter and hashed bag-of-words embedding are lightweight deterministic stand-ins. They avoid external dependencies; a production system could inject a real tokenizer and embedding function.

## 5. Part 4 — Security guardian

`SandboxGuardian` walks Python AST nodes and flags:

- Imports from `os`, `subprocess`, `socket`, `shutil`, `sys`, and `ctypes`.
- Calls to `eval`, `exec`, `compile`, `__import__`, and `open`.
- Access to `__subclasses__`, `__globals__`, `__builtins__`, and `__class__`.

`inspect_code()` also reports syntax errors. `audit_episodic_write()` rejects trust scores below 0.75, `UNTRUSTED_TELEMETRY` provenance, and text matching common privilege-escalation or safety-bypass directives.

**Security limitation:** AST inspection is a static screening layer, not a secure sandbox. It does not make arbitrary Python execution safe. Untrusted code should never be executed in the main application process.

## 6. How to run the modules

Use Python 3.11 or newer. From the project folder, run:

```bash
python rewoo_planner.py
python reflexion_surgeon.py
python memgpt_pager.py
python sandbox_guardian.py
```

Each file includes a small demonstration under its `if __name__ == "__main__":` block.

## 7. Manual verification checklist

- [ ] ReWOO blueprint lines are parsed into steps.
- [ ] Independent ReWOO steps appear in the same dependency level.
- [ ] Dependent steps execute after their prerequisites and receive substituted values.
- [ ] Token profiler returns all four required result keys.
- [ ] Traceback parser extracts a Python exception and location.
- [ ] Reflexion history retains no more than three entries.
- [ ] Duplicate candidate patches are rejected when the shared hash set is used.
- [ ] Memory pressure is reported at the configured threshold.
- [ ] FIFO items are consolidated into archival memory.
- [ ] Archival search ranks memories and returns the requested number.
- [ ] AST inspection flags the forbidden imports, calls, and attributes.
- [ ] Episodic writes below 0.75 or from untrusted telemetry are rejected.

## 8. Notes and limitations

- Tool functions for the ReWOO executor are supplied by the caller; no external tools or APIs are assumed.
- The repair loop accepts test and patch callbacks. It does not invent code patches by itself.
- The memory embedding and token count are educational approximations.
- AST policy checks are not a complete security boundary.
- The assignment brief does not specify a required command-line interface, test framework, or external LLM provider; this project therefore keeps those integrations injectable.

## 9. Submission

Before submission:

1. Run each module and inspect its output.
2. Complete the manual verification checklist and add any actual test results/screenshots required by your instructor.
3. Export this Markdown file as `mission-08-doc.pdf`.
4. Upload the PDF to the Week 08 Moodle Assignment Drop-box.
5. Submit the GitHub repository link in Tab 4 — “Transmit Solution” of the USSE Noosphere portal.
