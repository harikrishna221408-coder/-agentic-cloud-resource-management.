# Demonstration script (about 10 minutes)

Before the review: run `python -m src.app run` once, then `streamlit run dashboard/app.py` and open the page
it prints (usually http://localhost:8501). The first page load takes about 10 seconds; after that pages are quick.

## Storyline

> "Cloud machines cost money. Too little capacity slows applications; too much wastes money. Four agents each look at
> the problem from their own side, an orchestrator negotiates between them, and every recommendation is explained."

## Steps

1. **Overview** (sidebar: *Dashboard Overview*, strategy *Improved*).
   Show the 20 machines, the average current vs predicted CPU/memory, and the action counts. Point out the chart on
   the right: the three strategies give different answers on the same machines.
2. **Data Explorer.** Pick `m_694`. Show the CPU and memory charts and the heatmap. Say: *memory sits near 91% on every
   machine, so CPU is what separates busy from idle machines. That is why the memory alarm starts at 95%, not 85%.*
3. **Forecasting.** Show actual vs predicted for `m_694` and the findings box (it is computed from the metrics, so it always
   matches the table). Say plainly: *"The original notebook's features left out the current value. Adding the current CPU
   and memory cut the forest's error by about a fifth for CPU and almost a third for memory. Against the fair baseline,
   repeating the current value, the new model is slightly better for CPU (about 1.5% on MAE, 5% on RMSE) but not for memory,
   where simply repeating the last value still has the lower MAE. We report all four models."*
4. **Agent Monitoring** (machine `m_1223`, strategy *Improved*). Show the four cards. Resource says SCALE_UP
   (memory 95%), Performance and Cost say SCALE_DOWN (CPU 15%), and Security shows which policy checks failed.
   Say: *each agent has its own logic; they genuinely disagree.*
5. **Orchestration & Negotiation** (same machine). Show the conflict list and how the Improved strategy resolves it:
   MAINTAIN, because the scale-up signal is not corroborated and scaling down under memory pressure is unsafe.
   In the three-strategy row show **Original = SCALE_UP**, Improved = MAINTAIN, Cost-Aware = MAINTAIN.
6. **Explainability** (same machine). Show the full reasoning and the green "Checked" banner: the explanation is built
   from the record and automatically verified against it, and the decision traces back to its input row.
7. **Cost-Aware** (machine `m_694`). Orchestration page, switch strategy to *Cost-Aware*: Resource and Performance both
   want SCALE_UP, but it costs 6.40/h against a 6.00/h simulated budget, so the strategy keeps capacity and flags the
   decision for review. Say clearly: *the budget is a simulated assumption; with a 6.40 budget it matches Improved.*
8. **Agreement case** (machine `m_103`). The Resource, Performance and Cost Agents all vote SCALE_DOWN and Security approves; no conflicts; the cost saving is shown.
9. **Human approval.** Click *Approve (simulation)*. Say: *it only records the choice in the session; nothing in a real
   cloud is changed.*
10. **Evaluation.** Forecast tab (both Random Forests and both baselines, machines won), Coordinators tab (distribution and
    agreement), Threshold sensitivity (CPU threshold matters little, memory threshold matters a lot), Budget sensitivity
    (the Cost-Aware result flips at 6.40).

Terminal alternative: `python -m src.app analyze m_1223` prints the same forecast, decisions and explanations.
The Overview shows the *saved* run; machine pages recompute live from the latest observation, so counts can differ slightly.

## Likely questions and honest answers

**Is the Random Forest better than a simple baseline?**
Partly. The original notebook's features excluded the current value; that model was not better than simply repeating the
current value. We retrained with the current CPU and memory added: it is now slightly better than that baseline for CPU
(MAE 2.99 vs 3.04, RMSE 4.32 vs 4.56, better on 13 of 20 machines), but for memory persistence still has the lower MAE
(0.359 vs 0.412) while the forest has a slightly lower RMSE. The notebook-feature model is kept for comparison.

**Why did you retrain?**
While integrating, we found the original features left out the current observation, which also made the notebook's
baseline weak (it repeated an older value). Fixing that cut the Random Forest's error by 21% (CPU) and 30% (memory).

**Are the costs real?**
No. They are simulated (labelled `SIMULATED` everywhere). The budget and the 256 GB memory size are assumptions. The
Cost-Aware results depend on them.

**How do you know the decisions are correct?**
The trace has no "correct action" labels, so we can't measure accuracy. We verify that every decision is consistent with
the agents' calculations, traceable to its input, explained correctly, and that the system behaves sensibly (tested
scenarios with hand-computed answers, a randomised stress test, and monotonic behaviour as CPU rises).

**Why four agents and not one rule set?**
Each looks at a different concern (capacity, performance risk, cost, policy). Their disagreements are the point: the
orchestrator has to resolve them, and the explanation shows how.

**Why is the Security Agent not a cybersecurity detector?**
It checks resource-management policies (machine status, safe scale-down limits, unusual jumps, protected machines). It
does not detect intrusions or attacks, and the project does not claim to.

**Does it change anything in a real cloud?**
No. It is decision support only: no cloud credentials are needed, and every recommendation requires human approval.

**Why only 20 machines?**
The original experiment used the 20 machines with the most observations; we kept that. Results may not generalise.

**Why 95% for the memory threshold?**
Memory is about 91% on every machine in this trace (5th to 95th percentile: 84 to 96%), so the usual 85% limit would flag
every machine. The threshold is configurable and the sensitivity analysis shows how decisions change with it.

**What would you do next?**
Forecast further ahead with richer models, use verified prices, and evaluate the decisions by replaying the trace and
measuring outcomes.
