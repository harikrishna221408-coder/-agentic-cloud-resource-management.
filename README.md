# Agentic AI Orchestrated Cloud Resource Management
### Using Multi-Agent Negotiation and Explainable Decisions

A B.Tech AIML capstone project: a **decision-support system** that watches cloud machines, forecasts their CPU and
memory use, lets four specialised AI agents assess the situation, negotiates between them, and recommends one action
per machine (`SCALE_UP`, `SCALE_DOWN` or `MAINTAIN`) together with a plain-language explanation.

> **Safety:** this is a prototype. It never touches a real cloud resource, needs no cloud account, and every
> recommendation is marked *"human approval required"*. All **costs are simulated** (not real cloud prices).

---

## 1. The problem in simple words

Cloud machines cost money. If a machine has too little capacity, applications slow down. If it has too much, money is
wasted. Someone has to decide, for each machine, *add capacity, remove capacity, or leave it alone*, and that decision
involves competing concerns: performance, cost, and company rules.

**This project asks:** can several specialised agents, each looking at the problem from its own side, reach a
sensible joint decision, and can the system explain *why*?

## 2. Objectives

1. Analyse cloud workload and resource utilisation (Alibaba Cluster Trace 2018).
2. Forecast the next CPU and memory utilisation of each machine (Random Forest).
3. Assess each forecast with four agents: **Resource, Performance, Cost, Security**.
4. Coordinate the agents with a central **orchestrator** using three **negotiation strategies**.
5. Give every decision a human-readable, checkable **explanation**.
6. Evaluate the forecasts and the decisions honestly.
7. Present everything in an interactive **dashboard**.

## 3. How it works (architecture)

```
 Alibaba trace  ->  preprocessing  ->  Random Forest forecast (CPU, memory)
 (raw CSV)          (clean, lags)       |
                                        v  current + predicted CPU/memory
              +---------------------------------------------------------+
              |                    ORCHESTRATOR                         |
              |   validates input, calls the agents, checks their       |
              |   outputs, finds conflicts, runs a coordinator          |
              +--------+------------+-----------+------------+----------+
                       |            |           |            |
                 Resource      Performance     Cost       Security
                  Agent          Agent        Agent        Agent
                       \\            |           |            /
                        v           v           v           v
                 COORDINATOR  (Original | Improved | Cost-Aware)
                                    |
                                    v
                 final action  +  explanation  +  decision record (logged)
                                    |
                                    v
                      Streamlit dashboard  (human approves or rejects)
```

How the modules connect:

| Step | Code | Produces |
|---|---|---|
| Load + clean + features | `src/data/` | `data/processed/final_forecasting_dataset.csv` |
| Train + evaluate forecasts | `src/forecasting/` | `models/*.joblib`, `results/metrics/*` |
| Four agents | `src/agents/` | one assessment each |
| Orchestrator + negotiation | `src/orchestration/` | decision records |
| Explanations + checks | `src/explainability/`, `decision_log.py` | explanation text, audit log |
| Decision evaluation | `src/evaluation/` | sensitivity tables, graphs |
| One command for everything | `src/app.py` | all of the above |
| Dashboard | `dashboard/app.py` | the web interface |

## 4. Dataset

**Alibaba Cluster Trace 2018** (`machine_usage` and `machine_meta`), the production cluster trace published by
Alibaba (<https://github.com/alibaba/clusterdata>). Only this dataset is used.

| Fact | Value |
|---|---|
| Machines in the trace | 4,023 |
| Usage rows | 246,934,820 (8 days, a sample about every 10 seconds) |
| Machines used here | the **20 machines with the most observations** (1,290,470 rows) |
| Columns used | machine id, timestamp, CPU %, memory % (plus machine metadata: core count, status) |
| Columns not used | `mem_gps` and `mkpi` (about 80% missing), network and disk columns |

The files have **no header row**, so column names are assigned in `src/data/loader.py`. Raw data is not included
in this repository (about 8.6 GB); `scripts/download_data.py` downloads it.

> An earlier experiment used a FOCUS cloud *billing* sample. It has no CPU or memory data and is incompatible with the
> trace, so it is **not** used or mixed in anywhere.

### Preprocessing (`src/data/`)

1. Count observations per machine and keep the top 20.
2. Drop duplicate `(machine, time)` rows and rows with missing or out-of-range (outside 0-100) CPU or memory. *(On this
   data nothing needed removing: 0 duplicates, 0 invalid.)*
3. Sort by machine and time; compute the time gap to the previous and next observation.
4. Build the features and targets. The original notebook's **seven features** are CPU lag 1-3, memory lag 1-3 and the input time gap; the
   targets are the next CPU and memory value. A second feature set adds the **current CPU and memory** (9 features, see section 5).
5. Keep rows whose time gaps are both <= 60 s (1,287,777 rows remain).
6. **Chronological 80:20 split per machine** (first 80% of time to train, last 20% to test), so the model never sees the future.

## 5. Forecasting with Random Forest (`src/forecasting/`)

Random Forest regressors (100 trees, depth 15, seed 42), one for CPU and one for memory, trained on 1,030,214 rows and
tested on 257,563 unseen rows. **Two variants are trained with identical settings; they differ only in their inputs:**

| Variant | Inputs | Used for |
|---|---|---|
| `notebook` | the original 7 features: CPU lag 1-3, memory lag 1-3, input gap (lag 1 is the observation *before* the current one, so the current value is **not** an input) | comparison, reproducing the notebook |
| `current_value` | the same 7 features **plus the current CPU and memory** (9 features) | **the agents, the single-machine workflow and the dashboard** (`config/config.yaml: forecast.active_variant`) |

Models are saved with Joblib in `models/<variant>/`. `predict_next()` is the reusable prediction function: pass the
latest observations oldest first, *ending with the current one* (at least 4 values).

### Results (test set, error in percentage points; lower is better)

| Target | Model | MAE | RMSE |
|---|---|---|---|
| CPU | Random Forest (notebook features) | 3.793 | 5.424 |
| CPU | **Random Forest + current value** | **2.993** | **4.324** |
| CPU | Naive Persistence (original notebook baseline, repeats lag 1) | 3.839 | 5.694 |
| CPU | Persistence (current value), the standard one-step baseline | 3.038 | 4.559 |
| Memory | Random Forest (notebook features) | 0.586 | 0.847 |
| Memory | Random Forest + current value | 0.412 | **0.642** |
| Memory | Naive Persistence (original notebook baseline, repeats lag 1) | 0.532 | 0.859 |
| Memory | **Persistence (current value)** | **0.359** | 0.655 |

### Read this honestly

* **Adding the current value matters a lot:** it cuts the Random Forest's error by 21% for CPU and 30% for memory compared
  with the notebook's features, and it beats the notebook's baseline on all 20 machines for both resources.
* **Against the fair baseline (repeat the current value) the gain is small.** For CPU the new model is better, by 1.5% on
  MAE and 5.2% on RMSE (lower MAE on 13 of 20 machines): statistically clear on 257,563 rows, but modest in practice.
* **For memory, simple persistence still has the lower MAE** (the forest is 14.9% worse on MAE, 1.9% better on RMSE,
  and wins on 0 of 20 machines). Memory barely changes between readings, so repeating the last value is hard to beat.
* The notebook-feature model is **not better than persistence** on either resource (it wins CPU on 1 of 20 machines).
  The notebook's baseline was weak because it repeated lag 1, the observation *before* the current one.

## 6. The four agents (`src/agents/`)

Each agent has its own logic, its own input checks and its own output. All thresholds live in `config/agents.yaml`.
They were chosen from the data: memory sits near 91% on *every* machine, so a typical "85% memory" alarm would flag
everything; memory pressure therefore starts at 95%.

| Agent | Question it answers | Logic (defaults) |
|---|---|---|
| **Resource** | Is capacity short or idle? | Uses the higher of current and predicted. Pressure if CPU >= 80% or memory >= 95%. Over-allocated only if current **and** predicted CPU < 25%. |
| **Performance** | What could hurt performance? | Adds the forecast error (the active model's RMSE: 4.32 CPU, 0.64 memory) to the prediction, rates risk LOW/MEDIUM/HIGH, and checks what utilisation would be **after** adding or removing 25% capacity. |
| **Cost** | What would each action cost? | Hourly cost = vCPUs x price + memory GB x price, for scale-up, scale-down and maintain. Reports waste and whether a scale-up fits the budget. Never asks for more spend. |
| **Security** | Which actions do our policies allow? | Six policy checks (machine status, step size, memory guard, CPU guard on scale-down, unusual jump, protected machines) and returns the *allowed actions*. |

**About the Security Agent:** it checks resource-management *policies*. It is **not** a cybersecurity threat detector.

**About prices:** every price, the 256 GB memory size, and the 6.00 per machine-hour budget are **simulated
assumptions** (`pricing_source: SIMULATED`), not real cloud prices. Change them in `config/agents.yaml`.

## 7. Orchestration and negotiation (`src/orchestration/`)

The orchestrator, per machine: validates the input, calls the four agents (a crash or malformed answer from one agent
is isolated and recorded, never fatal), detects conflicts, runs the chosen coordinator, builds the explanation and saves
a decision record.

| Strategy | Rule in one paragraph |
|---|---|
| **Original** (from the notebook) | Follow any scale-up vote from Resource or Performance; scale down only if both agree; ignore cost. Security's restrictions are enforced. |
| **Improved** (conflict-aware) | Priority: Security, then Performance/Resource protection, then cost. A scale-up needs corroboration: Performance reports high risk, or Resource's pressure is confirmed by at least MEDIUM Performance risk. If Resource sees pressure but Performance sees low risk, that conflict is *not* resolved by scaling either way: **MAINTAIN**. A scale-down needs Performance's "safe" plus one supporter, and no missing agent. |
| **Cost-Aware** | The Improved decision, then the budget: an over-budget scale-up becomes MAINTAIN unless the forecast is critical (upper bound of CPU >= 90% or memory >= 98%); no new spend if the budget cannot be checked; if current spend already exceeds the budget and a scale-down is safe, scale down. |

Safe defaults: invalid input gives MAINTAIN flagged for review; a missing Security Agent gives MAINTAIN (policy cannot be
verified); a missing agent is never silently ignored.

### Six tested scenarios (`src/orchestration/scenarios.py`)

Expected outcomes were worked out by hand from the rules *before* running the code.

| Scenario | Original | Improved | Cost-Aware |
|---|---|---|---|
| High resource demand | SCALE_UP | SCALE_UP | SCALE_UP (critical forecast overrides the budget) |
| Low resource demand | SCALE_DOWN | SCALE_DOWN | SCALE_DOWN |
| High memory pressure | SCALE_UP | SCALE_UP | MAINTAIN (over budget, not critical) |
| Cost constraint violation | SCALE_UP | SCALE_UP | MAINTAIN |
| Security policy violation | MAINTAIN | MAINTAIN | MAINTAIN |
| Conflicting agents | SCALE_UP | MAINTAIN | MAINTAIN |

## 8. Explainability (`src/explainability/`)

Every decision has an explanation built **only from the decision record** (the real agent outputs), never from a fixed
template per action. It contains: the inputs, each agent's vote and reasoning, conflicts, how the decision was reached,
recommendations that were **not** followed (and why), the cost impact of the chosen action, the security checks, and the
human-approval statement.

Two safeguards: `check_explanation()` tests an explanation against its record (final action, every vote, cost figures,
budget status, security counts), and `verify_trace()` proves the decision traces back to its original input row. All
saved decisions pass both. Decisions are also written to an append-only log
(`results/decisions/decision_log.jsonl`).

## 9. Results on the real data

Saved run: each machine's most recent test-period observation (20 machines), forecast with the `current_value` model.

| Action | Original | Improved | Cost-Aware |
|---|---|---|---|
| SCALE_UP | 8 | 6 | 0 |
| SCALE_DOWN | 4 | 6 | 6 |
| MAINTAIN | 8 | 8 | 14 |

On a sample of 2,000 observations spread across the test period: SCALE_UP 323 / 192 / 9, SCALE_DOWN 142 / 365 / 365,
MAINTAIN 1,535 / 1,443 / 1,626 (Original / Improved / Cost-Aware). Neighbouring observations of one machine are
correlated, so this shows behaviour over time, not 2,000 independent samples.

* **Threshold sensitivity (Improved):** the CPU threshold matters little (SCALE_UPs: 201 at 70% down to 188 at 90%, because
  CPU is rarely above 70% here); the memory threshold matters a lot (199 at 93%, 192 at 95%, 99 at 97%).
* **Cost-Aware refuses nearly every scale-up** (0 of 6 on the saved run, 9 of the 2,000-observation sample). This depends
  **almost entirely on the simulated budget**: a scale-up costs 6.40 per hour against a 6.00 cap, and only a critical
  forecast overrides it. With a budget of 6.40 or more, Cost-Aware matches Improved (192). Treat this as a demonstration
  of how the strategy works, not a finding about real cloud costs.
* The notebook's original coordinator marked all 20 machines SCALE_UP, because its 85% memory limit flagged every machine.
* Agreement on the 20 machines: Original-Improved 80%, Improved-Cost-Aware 70%, Original-Cost-Aware 50%.

## 10. Installation

Requirements: Python 3.13 (tested), about 15 GB free disk for the raw data (only needed to reproduce from scratch; the download needs temporary space).

```bash
cd agentic-cloud-project
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## 11. How to run

```bash
# 1. (once) download the Alibaba trace, resumable, uses parallel connections. About 1.7 GB download, 8.6 GB extracted.
python scripts/download_data.py

# 2. run everything: data -> models -> agents -> 3 coordinators -> evaluation   (about 10 minutes the first time, mostly reading the 8.6 GB file; about 3.5 minutes with --skip-data)
python -m src.app run
#    reuse earlier stages:   --skip-data     or     --skip-data --skip-train   (about 1.5 minutes)

# 3. analyse one machine from the terminal
python -m src.app machines
python -m src.app analyze m_103

# 4. open the dashboard
streamlit run dashboard/app.py

# 5. run the automated tests
python -m pytest tests -q
```

Note: the download server is slow per connection; the downloader uses parallel range requests and took roughly
40 minutes on the development machine. If a part fails, run the script again; it resumes.

## 12. Dashboard

| Page | What you can do |
|---|---|
| Dashboard Overview | Machines analysed, average current/predicted CPU and memory, action counts, per-machine table |
| Data Explorer | Dataset preview, pick a machine and time window, CPU and memory charts, workload heatmap |
| Forecasting | Actual vs predicted for a machine, MAE/RMSE, baseline comparison |
| Agent Monitoring | Four agent cards (status, recommendation, risk, metrics, reasoning) |
| Orchestration & Negotiation | The four outputs, conflicts, final decision, all three strategies side by side, simulated approve/reject |
| Explainability | Full reasoning, cost and security assessment, automatic verification |
| Evaluation | Forecast metrics, decision distributions, agreement, threshold and budget sensitivity |

The sidebar chooses the page, the coordinator strategy, the machine and the observation. The **Overview** shows the saved
run; the machine pages recompute live from the machine's very latest observation, so their counts can differ slightly
(for example 10 instead of 8 Original scale-ups).

### Screenshots (captured from the running dashboard in Chrome, 1500 px wide)

| Page | Image |
|---|---|
| Overview | `docs/screenshots/01_overview.png` |
| Data Explorer | `docs/screenshots/02_data_explorer.png` |
| Forecasting | `docs/screenshots/03_forecasting.png` |
| Agent Monitoring | `docs/screenshots/04_agent_monitoring.png` |
| Orchestration & Negotiation | `docs/screenshots/05_orchestration.png` |
| Explainability | `docs/screenshots/06_explainability.png` |
| Evaluation | `07_evaluation_forecast.png`, `07b_evaluation_coordinators.png`, `08_evaluation_sensitivity.png`, `08b_evaluation_budget.png` |

They show the machine `m_1223` (a case where the agents disagree) with the *Improved* strategy; regenerate them after any
change to the dashboard or the results.

## 13. Testing

307 automated tests (`python -m pytest tests -q`), about 98% line coverage of `src/` and `dashboard/`:

| File | Covers |
|---|---|
| `test_data.py` | cleaning, lag features, no cross-machine leakage, gap filter, chronological split |
| `test_forecasting.py` | metrics, both Random Forests and both baselines, computed findings, prediction function, feature design (the new model reacts to the current value, the notebook model does not) |
| `test_agents.py` | each agent's logic with hand-computed expectations, invalid input, agent independence |
| `test_orchestrator.py` | decision records, crashing/malformed/missing agents, batches, duplicates, persistence |
| `test_coordinators.py` | the six scenarios x three strategies, budget rules, strategy comparison |
| `test_explainability.py` | explanation content, tamper detection, decision log, traceability |
| `test_integration.py` | the whole pipeline on a small synthetic trace, single-machine workflow, evaluation, loaders |
| `test_dashboard.py` | all seven pages rendered headlessly, strategy/machine selection, missing-results message |
| `test_robustness.py` | malformed input table, randomised invariants, determinism, checks on the real results |

Bugs the tests found and that were fixed are listed in `DEVELOPMENT_STATUS.md`. The dashboard has been tested
headlessly (Streamlit `AppTest`, on synthetic and real data) **and driven in real Chrome**: all pages and tabs were visited
and screenshotted with no error boxes and no browser console errors, and the screenshots were inspected page by page
(which found and fixed five layout problems the automated tests could not see: a clipped decisions table, mis-coloured
change arrows, truncated reason text, an unlabelled column and a clipped summary table).

## 14. Configuration

* `config/config.yaml`: dataset paths, number of machines, time-gap limit, train fraction.
* `config/agents.yaml`: every agent threshold, the simulated prices and budget, the security policies, the scaling step
  (25%), and the Cost-Aware critical limits. Nothing is hard-coded in the agents.

## 15. Project structure

```
config/              settings (data, agents, pricing, policies)
dashboard/           Streamlit app + chart builders
scripts/             download_data.py
src/data/            loader, preprocessing, pipeline
src/forecasting/     train, predict, evaluation
src/agents/          resource, performance, cost, security (+ shared validation)
src/orchestration/   orchestrator, negotiation helpers, coordinators, scenarios, decision log
src/explainability/  explanation builder + consistency checker
src/evaluation/      decision evaluation, sensitivity analysis
src/app.py           one-command pipeline and CLI
src/service.py       single-machine workflow
src/dashboard_data.py  loaders used by the dashboard
tests/               307 tests
results/             metrics, graphs, decision records (generated)
models/              saved Random Forest models, one folder per variant (generated)
data/                raw and processed data (generated, not in git)
DEVELOPMENT_STATUS.md  progress log, findings and known limitations
LICENSE              MIT licence for the code (the dataset is not covered)
docs/DEMO.md         10-minute demonstration script and likely panel questions
```

## 16. Differences from the original notebook

* The notebook's logic was moved into tested modules; the four agents now make independent recommendations and the
  coordinators combine the agents' own votes, instead of re-computing thresholds inside the coordinator.
* The notebook's duplicated `input_gap` column became one `input_gap` and one `target_gap`; final dataset 1,287,777 rows
  vs 1,287,817 in the notebook (40 rows, 0.003%, from the final gap filter; the exact cause was not found).
* Memory pressure threshold 85% -> 95% (the data sits at about 91%); cost uses an explicit simulated allocation model.
* Added the fair persistence baseline, validation, explanations with self-checks, logging, configuration and tests.
* **Retrained with the current CPU and memory as extra features** (the notebook's features left the current value out), keeping the
  notebook-feature models for comparison. The Performance Agent's error margins were updated to the new model's RMSE.

## 17. Limitations

* **The forecast is only a little better than repeating the current value** (section 5): slightly better for CPU, and
  worse on MAE for memory. The agents' value lies mainly in the decision logic, not in a superior forecast.
* **Costs, memory size and budget are simulated assumptions;** the Cost Agent and the Cost-Aware results depend on them.
* **Only 20 machines**, chosen for having the most observations, over 8 days. Most are lightly loaded (median CPU 38%),
  so SCALE_DOWN candidates are common and real overload is rare. Results may not generalise to other clusters.
* **No ground truth for decisions:** the trace has no "correct action" labels, so decisions are checked for
  consistency, traceability and sensible behaviour, not for correctness against an oracle.
* **A single fixed 25% scaling step;** no instance catalogue, no migration or startup delay, no multi-machine capacity.
* **Rule-based negotiation,** thresholds set by hand from the data distribution; not learned.
* **The Security Agent checks resource policies only,** not threats.
* **Dashboard checked only in desktop Chrome at 1500 px width** (not on phones, small windows or other browsers); no authentication, single user.

## 18. Future scope

* Try richer models (gradient boosting, LSTM) and more history features; memory in particular needs a better signal than the last value.
* Forecast several steps ahead and add uncertainty intervals to the agents.
* Replace simulated prices with verified provider pricing and a real instance catalogue.
* Learn the negotiation (reinforcement learning with a defined reward) and compare with the hand-written strategies.
* Add a dry-run integration with a cloud provider (read-only first; never automatic changes without approval).
* Evaluate on more machines, other traces and with a replay simulator that measures the *outcome* of decisions.

## 19. Licence, data and acknowledgement

**Code:** released under the MIT licence; see the [`LICENSE`](LICENSE) file. This is a student capstone prototype and
comes with no warranty.

**Data:** the Alibaba Cluster Trace 2018 is **not** part of this repository and is **not** covered by the MIT licence.
It is provided by Alibaba for research and keeps its own terms; see the trace repository
(<https://github.com/alibaba/clusterdata>) for those terms and for how to cite it.
