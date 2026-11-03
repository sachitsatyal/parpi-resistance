"""Per-dose summary of held-out (cross-validated) predictions."""

import pandas as pd

cv = pd.read_csv("outputs/metrics/cv_predictions.csv")
mean = cv.pivot_table(index="dose_um", columns="model", values="pred", aggfunc="mean")
mean.insert(0, "target", cv.groupby("dose_um")["target"].first())
print("Mean predicted score per dose (held-out), target = true position:")
print(mean.round(3).to_string())
print("\nRank of each dose by mean prediction (1 = lowest; should be 1..10 in order):")
print(mean.drop(columns="target").rank().astype(int).to_string())
print("\nMean uncertainty (score std) per dose:")
print(cv.pivot_table(index="dose_um", columns="model", values="pred_std").round(3).to_string())
