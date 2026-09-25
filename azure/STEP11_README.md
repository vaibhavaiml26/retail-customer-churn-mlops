# Step 11 - Azure ML production pipeline

This step wraps the already validated Step 9 monthly scorer and Step 10
champion/challenger retrainer in one Azure ML SDK v2 pipeline.

## Files

- `submit_pipeline_job.py` - defines and submits the Azure ML pipeline.
- `retrain_azure.py` - Step 10 retrainer with one important production fix:
  it skips retraining when the current champion already consumed the newest
  fully-resolved snapshot.
- `.amlignore.step11.example` - merge these exclusions into the `.amlignore`
  in the project root to avoid uploading raw datasets, models, ZIPs, and other
  large local artifacts as source code.

Keep your existing `azure/score_azure.py` from Step 9.

## Install

Copy `retrain_azure.py` over your existing `azure/retrain_azure.py` and put
`submit_pipeline_job.py` in the same `azure/` folder.

Project layout:

```
Production Design_GPT1.1/
  config.py
  data_loader.py
  data_validation.py
  feature_engineering.py
  models.py
  .amlignore
  azure/
    score_azure.py
    retrain_azure.py
    submit_pipeline_job.py
```

Fill the same subscription/resource-group/workspace values in
`submit_pipeline_job.py` that you used in Steps 9 and 10.

## First validation run

The file is configured for:

```
AS_OF_DATE = "2011-10-01"
```

Expected behavior:

1. `monthly_score` uses the latest registered champion (currently v2) and
   scores customers using history strictly before 2011-10-01.
2. `retrain_check` reconstructs the same point-in-time history.
3. Only five labeled snapshots are fully resolved at this date, and champion
   v2 metadata says it already consumed snapshot 5.
4. Retraining therefore exits with `retrain_skipped=true` rather than training
   another duplicate challenger.
5. No new model version is registered.

Run from the project root:

```powershell
python .\azure\submit_pipeline_job.py
```

## Later historical replay

Change only `AS_OF_DATE` for each monthly replay:

- `2011-10-01` -> score October, retrain should skip
- `2011-11-01` -> score November, retrain should still skip
- `2011-12-01` -> score December, snapshot 6 is now fully resolved and the
  retraining branch should train the next challenger using rolling snapshots
  `[3,4] | [5] | [6]`

The submitter registers a challenger only when `promotion_decision.json`
contains `promote=true`. Rejected or skipped challengers never become model
assets, so the latest registered model remains the operational champion.
