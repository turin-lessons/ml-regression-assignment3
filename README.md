# Assignment 3 — Think First, Then Code (Variant A)

Linear, logistic and softmax regression applied to software-engineering data
(code-review time, CI build time, deployment failures, incident triage).

## Two ways to check the work

| Option | What to open | When to use |
|---|---|---|
| **Notebook** | [`notebooks/Assignment3_VariantA.ipynb`](notebooks/Assignment3_VariantA.ipynb) | Full report: "Think first" reasoning, code, outputs, plots and comparisons in one place. Self-contained — runs in Jupyter or Google Colab with no other files. |
| **Python project** | `tasks/`, `src/`, `data/` | Same solutions as modular, runnable scripts with tests. |

## Project structure

```
.
├── data/                       # every dataset from the assignment, one file per question
│   ├── q1_code_review.csv
│   ├── q2_ci_build.csv
│   ├── q3_deployment.csv
│   ├── q4_single_sample.json
│   ├── q5_logits.csv
│   └── q6_incidents.csv
├── src/regression/             # reusable code, grouped by method
│   ├── data.py                 # CSV / JSON loaders
│   ├── linear.py               # least squares, feature map, MSE, R², gradient descent
│   ├── logistic.py             # sigmoid, BCE loss, one GD step, threshold evaluation
│   ├── softmax.py              # stable softmax, multiclass log loss
│   └── plotting.py             # saves figures to outputs/figures/
├── tasks/                      # one runnable script per question
│   ├── q1_code_review.py
│   ├── q2_ci_build.py
│   ├── q3_deployment_failure.py
│   ├── q4_logistic_update.py
│   ├── q5_softmax_outputs.py
│   └── q6_incident_classification.py
├── notebooks/Assignment3_VariantA.ipynb
├── outputs/figures/            # generated plots (created when scripts run)
├── tests/test_regression.py
├── run_all.py
└── requirements.txt
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                    # all six questions
python tasks/q3_deployment_failure.py  # a single question
pytest -q                            # tests
jupyter notebook notebooks/Assignment3_VariantA.ipynb
```

## Key results

| Q | Result |
|---|---|
| 1 | β₀ = 9.643, β₁ = 7.690, R² = 0.999; ŷ(5.5) = 51.940, ŷ(10) = 86.548 |
| 2 | MSE linear = 2.738 vs quadratic = 0.007; GD converges for α < 0.066 (β* = 4.173) |
| 3 | Both coefficients positive; P(FAIL \| [1.2, 0.8]) = 0.518; threshold 0.5 → acc 1.000, 0.3 → 2 false alarms, acc 0.750 |
| 4 | p: 0.537 → 0.619, loss: 0.621 → 0.480 after one step |
| 5 | argmax = Backend, Frontend, Database, Frontend; softmax invariant to adding a constant |
| 6 | Accuracy 1.000, log loss 0.552, W is 3 × 2 |
# ml-regression-assignment3
