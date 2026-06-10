import pandas as pd

sub1 = pd.read_csv("probabilities\\lgb_07_06_2026_v1.csv")
sub2 = pd.read_csv("probabilities\\lgb_07_06_2026_v2.csv")

weights = [0.96597, 0.96579]
w1 = weights[0] / sum(weights)
w2 = weights[1] / sum(weights)

prob_cols = ["GALAXY", "QSO", "STAR"]

blend = pd.DataFrame()
blend["id"] = sub1["id"]

# Blend probabilities
for col in prob_cols:
    blend[col] = w1 * sub1[col] + w2 * sub2[col]

# Final prediction as string
blend["class"] = blend[prob_cols].idxmax(axis=1)
submission = blend[["id", "class"]]
submission.to_csv("submissions\\blends\\blend_lgb_sub_10_06_2026_v1_lgb_sub_10_06_2026_v2.csv",index=False)