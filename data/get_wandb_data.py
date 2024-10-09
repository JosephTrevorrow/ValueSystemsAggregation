import wandb

api = wandb.Api()
runs = list(range(1, 11))

# print all available runs
wandb_dir = "joseph-trevorrow-university-of-bristol/valuesystemsaggregation"
runs = api.runs(wandb_dir)

filtered_runs = [run for run in runs if run.config["society"] == "norm_society"]

# Sanity Check
for run in filtered_runs:
    print(f"FILTERED Run ID: {run.id}, Name: {run.name}, State: {run.state}, Created: {run.created_at}")
    metrics_df = run.history()
    metrics_df.to_csv(f"/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/wandb_run_data/{run.name}.csv", index=False)