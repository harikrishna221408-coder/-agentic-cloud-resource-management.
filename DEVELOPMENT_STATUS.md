# Development Status

Dataset: Alibaba Cluster Trace 2018 (machine_usage, machine_meta). FOCUS (billing data) is NOT used.

- [x] Phase 1: Audit of existing notebook
- [x] Phase 2: Data loading + preprocessing (5 tests pass)
- [x] Phase 3: RF forecasting + baseline (10 tests pass in total)
- [x] Phase 4: Four agents (68 tests pass in total)
- [x] Phase 5: Orchestrator (90 tests pass in total)
- [x] Phase 6: Three coordinators / negotiation (123 tests pass in total)
- [x] Phase 7: Explainability + decision logging (182 tests pass in total)
- [x] Phase 8: Integration (210 tests pass in total)
- [x] Phase 9: Streamlit dashboard (229 tests pass in total)
- [x] Phase 10: Full testing (298 tests pass)
- [x] Phase 11: Documentation (README.md, docs/DEMO.md, docs/screenshots placeholders, pinned requirements.txt)

## Known differences from the original notebook
- Feature dataset: 1,287,777 rows vs notebook 1,287,817 (-40, 0.003%). All 1,290,390 complete rows match;
  the difference comes from the final time-gap filter (this project requires input gap AND target gap <= 60 s).
- Notebook's duplicated `input_gap` column replaced by one `input_gap` + one `target_gap`.
- Test set: 257,563 rows (notebook 257,571), same cause.

## Forecasting results (chronological 80:20, test set)
| Target | Model | MAE | RMSE |
|---|---|---|---|
| CPU | Random Forest | 3.7932 | 5.4239 |
| CPU | Naive Persistence | 3.8393 | 5.6940 |
| Memory | Random Forest | 0.5861 | 0.8473 |
| Memory | Naive Persistence | 0.5325 | 0.8591 |

RF beats this (notebook) baseline on CPU for 13/20 machines. On memory RF has worse MAE on 20/20 machines (slightly better RMSE overall).
**See the Phase 8 forecasting limitation below: this baseline is weak.**

## Phase 4: agent behaviour on each machine's latest real observation (20 machines)
| Agent | SCALE_UP | SCALE_DOWN | MAINTAIN |
|---|---|---|---|
| Resource | 8 | 5 | 7 |
| Performance | 2 | 14 | 4 |
| Cost | 0 (never asks for more spend) | 14 | 6 |
Security verdicts: 7 APPROVED, 13 RESTRICTED (SCALE_DOWN blocked by the memory guard). The agents disagree on many
machines (e.g. m_1223: Resource wants SCALE_UP for memory 95%, Performance and Cost want SCALE_DOWN for CPU 16%);
these conflicts are what the coordinators in Phase 5/6 must resolve.

## Assumptions to review (config/agents.yaml)
- Thresholds were derived from the data: memory is ~91% on every machine, so memory pressure starts at 95%.
- All prices are SIMULATED. Budget (6.0 per machine-hour) and simulated memory size (256 GB) are assumptions;
  the trace's mem_size is normalised (100), not GB. With these values SCALE_UP (6.40/h) exceeds the budget.
- Scaling step is 25% for every action.
- The Security Agent checks resource-management policies only; it is not a threat detector.

## Phase 5: orchestrator on the 20 real machines (Original coordinator)
SCALE_UP 8, SCALE_DOWN 3, MAINTAIN 9 (the notebook's original coordinator gave 20/20 SCALE_UP with an 85% memory limit).
20 unique decisions, 0 missing explanations. Records: results/decisions/decisions_original.jsonl / .csv.
Explanations are still the basic coordinator rationale; the full explainability module is Phase 7.
Safe defaults: invalid input -> MAINTAIN + review; missing Security Agent -> MAINTAIN + review;
other failed/malformed agent -> recorded in agent_errors, decision marked DEGRADED.

## Phase 6: coordinator strategies on the 20 real machines
| Action | Original | Improved | Cost-Aware |
|---|---|---|---|
| SCALE_UP | 8 | 6 | 0 |
| SCALE_DOWN | 3 | 6 | 6 |
| MAINTAIN | 9 | 8 | 14 |
All three agree on 9 of 20 machines. Files: results/decisions/decisions_<strategy>.jsonl/.csv, strategy_comparison.csv.
The six required scenarios (src/orchestration/scenarios.py) have expected outcomes worked out by hand per strategy and are tested.
- Improved differs from Original: it refuses uncorroborated scale-ups (2 machines) and accepts safe scale-downs that
  Original ignores (3 machines).
- Cost-Aware refuses ALL 6 scale-ups, because the SIMULATED budget (6.0/h) is below the cost of a scale-up (6.40/h) and no
  machine reaches the critical forecast limit (upper-bound CPU 90% / memory 98%). This result depends entirely on that
  budget assumption (config/agents.yaml) and should be tuned or justified before presenting it as a finding.
- Fixed during Phase 6: the Security Agent's requires_review no longer fires merely because SCALE_DOWN is restricted.

## Phase 7: explainability and decision logging
- src/explainability/explanation.py builds structured explanations (votes, agent reasoning, conflicts, decision path,
  recommendations not followed, cost impact, security checks, human-approval statement) only from the decision record.
- check_explanation() tests an explanation against its record (final action, votes, cost numbers, budget status,
  security check counts); verify_trace() checks a decision against its original input.
- src/orchestration/decision_log.py: append-only JSONL log with run ids (results/decisions/decision_log.jsonl).
- All 60 real records (3 strategies x 20 machines): 0 explanation inconsistencies, 0 trace failures.
- Bug found by reading real output and fixed: Cost-Aware listed "not followed" reasons relative to the Improved step
  and attributed its own budget refusal to the Cost Agent. Regression tests added.

## Phase 8: integration
- `python -m src.app run [--skip-data] [--skip-train]`: data -> models -> agents -> 3 coordinators -> evaluation
  (387 s on the full data with training; 0 explanation/trace problems). `python -m src.app analyze m_103` and
  `python -m src.app machines` use the single-machine workflow (src/service.py). Dashboard loaders: src/dashboard_data.py.
- Evaluation outputs in results/metrics: decision_distribution(.csv/_sample.csv), strategy_agreement, agent_votes,
  threshold_sensitivity, budget_sensitivity, run_summary.json; graphs in results/graphs.

### KNOWN FORECASTING LIMITATION (found in Phase 8, decision: keep the notebook's method, report honestly)
The 7 features use lag 1-3 = the observations BEFORE the current one, and the target is the one AFTER it, so the current
value is not a model input. The notebook's "Naive Persistence" baseline repeats lag 1 (stale). A standard one-step
persistence baseline (repeat the current value) is added and is better than Random Forest:
| Target | Random Forest | Naive Persistence (notebook, lag 1) | Persistence (current value) |
|---|---|---|---|
| CPU MAE / RMSE | 3.793 / 5.424 | 3.839 / 5.694 | **3.038 / 4.559** |
| Memory MAE / RMSE | 0.586 / 0.847 | 0.533 / 0.859 | **0.359 / 0.655** |
RF beats the current-value baseline on 1/20 machines for CPU and 0/20 for memory. Possible future fix: add the current
value as an 8th feature and retrain (not done; would change the reported results).

### Decision evaluation on a 2000-observation sample spread over the test period (autocorrelated, not independent)
| Action | Original | Improved | Cost-Aware |
|---|---|---|---|
| SCALE_UP | 332 | 190 | 2 |
| SCALE_DOWN | 113 | 342 | 342 |
| MAINTAIN | 1555 | 1468 | 1656 |
Pairwise agreement on the 20 latest observations: original-improved 75%, improved-cost_aware 70%, original-cost_aware 45%.
Threshold sensitivity (Improved): the CPU-high threshold barely matters (70 -> 90 changes SCALE_UP 191 -> 186, because CPU is
rarely above 70%); the memory-high threshold matters a lot (93 -> 194, 95 -> 190, 97 -> 100 SCALE_UPs).
Budget sensitivity (Cost-Aware): budgets of 6.0 or less give 2 scale-ups; 6.4 or more gives 190 (equal to Improved).

## Phase 9: dashboard
`streamlit run dashboard/app.py` (needs `python -m src.app run` first; shows a clear message if results are missing).
Pages: Overview, Data Explorer, Forecasting, Agent Monitoring (four agent cards), Orchestration & Negotiation (strategy
selector, conflicts, final decision, three-strategy comparison, simulated approval buttons), Explainability (with an
automatic explanation/trace check), Evaluation (forecast metrics, coordinators, threshold and budget sensitivity).
Tested with Streamlit's AppTest on a synthetic project AND on the real data (all 7 pages, 0 exceptions; first load ~11 s,
pages ~0.1-2 s afterwards); the real server was started and answered its health check.
(Visual check done later: see 'Browser check of the dashboard' below.)
Bug found by the tests and fixed: sliders crashed on machines with short histories.

## Phase 10: full testing and validation
- 298 tests, all passing: in one run, per file in isolation, and in reverse order (no order dependence).
- Line coverage of src/ and dashboard/: see the final number in the Phase 10 report (98% before the robustness tests).
- Randomised stress test (seeded; 4,500 decisions in exploration, 1,200+ per run in the suite): no crashes, no mutated
  input, the final action is always allowed by the Security Agent, every explanation and trace consistent, results are
  deterministic, decisions are monotonic in CPU, and all three actions are reached under every strategy.
- Bugs found and fixed in Phase 10 (input validation accepted bad data silently): booleans and numeric strings were
  accepted as numbers; empty/blank machine ids were accepted; cpu_num was never validated (NaN produced cost=nan, a negative
  value produced a nonsense cost, a string crashed the Cost Agent). All are now rejected with a clear message and a safe
  MAINTAIN decision flagged for review. Real-data results are unchanged.
- Acceptance checks on the real results: all 20 machines have exactly one explained, traceable decision per strategy;
  decisions trace to forecast rows; the saved forecast metrics match a fresh recomputation; the split is chronological.
- Environment note: one pytest run reported 5h40m wall-clock for 7 s of work; the machine had been suspended (sleep).

## Final acceptance criteria (spec section 16) - evidence
1 run locally: the Streamlit server started and passed its health check | 2 load dataset: Data Explorer, MachineStore
3 select a machine | 4 current CPU/memory | 5 predictions | 6 sent to four agents | 7 agent evaluations (Agent Monitoring)
8 three coordinators | 9 final recommendation | 10 reasoning (Explainability, auto-verified) | 11 graphs (Forecasting, Evaluation)
12 automated tests pass. Items 3-11 are covered by headless AppTest runs on synthetic and real data.
(Visual appearance later verified in Chrome: see 'Browser check of the dashboard'.)

## Retraining with the current value (after Phase 11, on request)
Decision: the original 7-feature model excluded the current observation. Both variants are now trained with identical settings:
`notebook` (7 features) and `current_value` (the same + current CPU and memory = 9 features). `current_value` is the active
variant (config/config.yaml: forecast.active_variant) used by the agents, single-machine workflow and dashboard; models are in
models/<variant>/. predict_next() now takes >= 4 observations ending with the CURRENT one.
| Target | RF notebook features (MAE / RMSE) | RF + current value | Persistence (current value) |
|---|---|---|---|
| CPU | 3.793 / 5.424 | **2.993 / 4.324** | 3.038 / 4.559 |
| Memory | 0.586 / 0.847 | 0.412 / **0.642** | **0.359** / 0.655 |
- New model vs old: -21.1% MAE (CPU), -29.6% MAE (memory). Beats the notebook's lag-1 baseline on 20/20 machines (both resources).
- New model vs persistence: CPU better (MAE -1.5%, RMSE -5.2%; 13/20 machines; paired difference clearly non-zero on 257,563 rows);
  memory worse on MAE (+14.9%, 0/20 machines), slightly better on RMSE (-1.9%).
- The Performance Agent's margins were updated to the new RMSE (CPU 4.32, memory 0.64); a test now checks they match the saved results.
- Decisions on the saved run (20 machines): Original 8/4/8, Improved 6/6/8, Cost-Aware 0/6/14 (SCALE_UP/SCALE_DOWN/MAINTAIN).
  2,000-sample: SCALE_UP 323/192/9, SCALE_DOWN 142/365/365, MAINTAIN 1,535/1,443/1,626. All 305 tests pass; 0 explanation/trace problems.
- The dashboard's forecasting findings are now computed from the metrics (evaluation.verdict) instead of hard-coded text.
- Found and fixed while doing this: predict_next raised a bare KeyError for an unknown variant; read_log/load_decisions leaked file
  handles; verdict() divided by zero for a perfect baseline. The real-results trace test correctly failed on the stale saved
  decisions until the pipeline was re-run.
- Supersedes the earlier "KNOWN FORECASTING LIMITATION" and Phase 8/9/10 forecasting figures above (kept as history).

## Browser check of the dashboard (after Phase 11, on request)
No browser tool was available in the session, so the running dashboard (`streamlit run`, port 8610) was driven in real desktop
Chrome with Playwright (installed only for this check, not in requirements.txt) at 1500 px width. All 7 pages and the 4
Evaluation tabs were visited: 0 Streamlit error boxes, 0 browser console errors. Screenshots (docs/screenshots/, cropped) were
inspected one by one. Overall: renders correctly and looks clean. Five layout problems were found and fixed:
1. Overview: the decisions table was cut off on the right, hiding final_action (now full width; chart moved below).
2. Delta arrows used Streamlit's default colours (up = green); for utilisation, rising is the bad direction. Now inverted, and
   changes under 0.05 show no badge (previously "-0.0 pts" in red and "+0.00" cost in red).
3. Long reason/detail text was truncated in tables (now st.table, which wraps).
4. The "followed" column was unlabelled checkboxes (now "final decision consistent with it?" yes/no).
5. The "machines won" table was one wide row, clipped (now a two-column table).
Regression tests added (delta colours, final_action visible). 307 tests pass. Limits: desktop Chrome at 1500 px only; not
tested on phones, small windows or other browsers; Agent Monitoring cards have slightly uneven heights (cosmetic, left as is).
