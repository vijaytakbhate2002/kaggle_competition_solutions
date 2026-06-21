import pandas as pd

sub1 = pd.read_csv("probabilities\\lgb_14_06_2026_v1.csv")
sub2 = pd.read_csv("probabilities\\lgb_14_06_2026_new_features_v1.csv")
sub3 = pd.read_csv("probabilities\\lgb_14_06_2026_v2.csv")
sub4 = pd.read_csv("probabilities\\lgb_16_06_2026_all_features_v1.csv")
sub5 = pd.read_csv("probabilities\\lgb_19_06_2026_new_features_v1.csv")

weights = [0.96597, 0.96545, 0.95946, 0.96696, 0.9632]
w1 = weights[0] / sum(weights)
w2 = weights[1] / sum(weights)
w3 = weights[2] / sum(weights)
w4 = weights[3] / sum(weights)
w5 = weights[4] / sum(weights)

prob_cols = ["GALAXY", "QSO", "STAR"]

blend = pd.DataFrame()
blend["id"] = sub1["id"]

# Blend probabilities
for col in prob_cols:
    blend[col] = (
        w1 * sub1[col] + 
        w2 * sub2[col] +
        w3 * sub3[col] +
        w4 * sub4[col] +
        w5 * sub5[col]
    )

# Final prediction as string
blend["class"] = blend[prob_cols].idxmax(axis=1)
submission = blend[["id", "class"]]
submission.to_csv("submissions\\blends\\blend_lgb_top_5_solutions_blend.csv",index=False)




