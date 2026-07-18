# MLS Lab 1 — Linear Regression (Autograded)

Autograded lab: students implement linear regression with batch gradient
descent from scratch, submit via a GitHub PR, and get an automatic score
comment from a GitHub Actions bot.

## How it works

```
Student pushes/opens PR
        │
        ▼
GitHub Actions runs (.github/workflows/autograde.yml)
        │
        ├─► Convert notebook → script, execute it
        │      (defines student's functions + trains + saves outputs/theta.npy)
        │
        ├─► pytest grading/test_student_lab.py   → correctness score (60%)
        │
        ├─► grading/grade_performance.py         → performance score (40%)
        │      (loads theta.npy, evaluates on the HIDDEN test set)
        │
        └─► grading/combine_and_post.py → posts final score as a PR comment
```

## One-time instructor setup

1. **Generate the datasets** (do this once, locally, before creating the
   template repo):
   ```bash
   cd data
   python generate_data.py
   ```
   This creates `data/train.csv` (ships to students) and
   `instructor_private/test_hidden.csv` (must NEVER be committed to the
   student-facing repo — see `.gitignore`).

2. **Create the student template repo** from everything in this folder
   *except* `instructor_private/`. GitHub Classroom or a plain template
   repo both work fine.

3. **Make the hidden test set available to the grading workflow without
   exposing it to students.** Pick one:
   - **Simplest:** base64-encode `test_hidden.csv` and store it as a repo
     secret named `HIDDEN_TEST_CSV_B64` (`base64 -w0 test_hidden.csv`).
     The workflow already decodes this in the "Fetch private hidden test
     set" step.
   - **Fits your Kaggle workflow:** upload it as a private Kaggle dataset
     and pull it in CI with `kaggle datasets download` using
     `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets.
   - **Most robust for many labs:** keep a separate private
     `mls-grading-data` repo and clone it in CI with a fine-grained PAT
     stored as a secret.

4. **Add the secret(s)** under repo Settings → Secrets and variables →
   Actions. If using GitHub Classroom, add them at the organization level
   so every student repo created from the template inherits them.

## What students receive

- `student_lab_template.ipynb` — the notebook with TODO cells
- `data/train.csv` — training data
- `.github/workflows/autograde.yml` — runs automatically, no setup needed
  on their end

They should **not** receive `instructor_private/`, `grading/`, or this
README's setup section (or at least shouldn't need to touch them) —
feel free to trim the README down to just the "Lab instructions" you add
on top before handing it out.

## Adjusting the weighting or grade bands

- Correctness/performance split: `CORRECTNESS_WEIGHT` /
  `PERFORMANCE_WEIGHT` in `grading/combine_and_post.py` (currently 60/40).
- Performance → score bands (R² thresholds): `score_from_r2()` in
  `grading/grade_performance.py`.
- Add/remove correctness checks: `grading/test_student_lab.py`. Each
  `test_*` function is one graded check.

## Local testing (before assigning to students)

Fill in the TODOs yourself in a copy of the notebook, then run the same
steps CI runs:

```bash
jupyter nbconvert --to script student_lab_template.ipynb --output student_solution
python student_solution.py
pytest grading/test_student_lab.py --junitxml=results.xml
python grading/grade_performance.py
```

This whole pipeline was tested end-to-end with a correct reference
solution: 7/7 correctness tests passed, R² = 0.99 on the hidden set →
100/100 final score.
