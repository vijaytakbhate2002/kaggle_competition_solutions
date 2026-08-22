"""
Patch smart_ensembling.ipynb:
  1. Step 3 config cell  -> add RUN_TIMESTAMP / RUN_DIR timestamped subdir
  2. Step 6d save cell   -> route ALL artifacts (npy, csv, json) to RUN_DIR
  3. SHAP/importance cell (c8d13307) -> fill with perm-importance + SHAP + plot
"""
import json
from pathlib import Path

NB_PATH = Path("smart_ensembling.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ── helpers ──────────────────────────────────────────────────────────────────
def find_cell(nb, cell_id):
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == cell_id:
            return i, c
    return None, None

def src(*lines):
    """Return a list of source lines with trailing \\n (last line has no \\n)."""
    out = []
    for j, line in enumerate(lines):
        out.append(line + "\n" if j < len(lines) - 1 else line)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 1. Step 3 — add RUN_TIMESTAMP + RUN_DIR
# ─────────────────────────────────────────────────────────────────────────────
idx3, cell3 = find_cell(nb, "cell_step3")
assert cell3 is not None, "cell_step3 not found"

new_src3 = src(
    "# \u2500\u2500 Configuration (edit paths here if needed) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    'PREDICTION_ROOT = Path("oof_&_probs")',
    'TRAIN_CSV       = Path("input_data/train.csv")',
    'TEST_CSV        = Path("input_data/test.csv")',
    'TARGET_COL      = "addicted_label"',
    "",
    "# \u2500\u2500 Timestamped run directory \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# Every execution gets its own folder so all experiments are preserved.",
    'ARTIFACTS_ROOT  = Path("ensemble_artifacts")',
    'RUN_TIMESTAMP   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")',
    'RUN_DIR         = ARTIFACTS_ROOT / f"{RUN_TIMESTAMP}_hill_climb"',
    "ARTIFACTS_DIR   = ARTIFACTS_ROOT   # backward-compat alias",
    "",
    "# Manifest stays at root (shared across runs; describes file layout)",
    'MANIFEST_PATH   = ARTIFACTS_ROOT / "paired_prediction_paths.json"',
    "",
    "ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)",
    "RUN_DIR.mkdir(parents=True, exist_ok=True)",
    'print(f"Run directory  : {RUN_DIR}")',
    'print(f"Manifest       : {MANIFEST_PATH}")',
    "",
    "",
    "def rel(path: Path, root: Path) -> str:",
    "    return path.relative_to(root).as_posix()",
    "",
    "",
    "def find_prediction_pairs(root: Path) -> dict:",
    "    # Supports: oof_<name>.npy <-> test_<name>.npy",
    "    #           <name>_oof.npy <-> <name>_test.npy",
    "    pairs = {}",
    '    for oof_path in sorted(root.rglob("*.npy")):',
    "        name = oof_path.name",
    '        if name.startswith("oof_"):',
    "            test_name = f\"test_{name[len('oof_'):]}\"",
    '        elif name.endswith("_oof.npy"):',
    "            test_name = f\"{name[:-len('_oof.npy')]}_test.npy\"",
    "        else:",
    "            continue",
    "        test_path = oof_path.with_name(test_name)",
    "        if test_path.is_file():",
    "            pairs[rel(oof_path, root)] = rel(test_path, root)",
    "        else:",
    "            print(",
    "                f\"[UNPAIRED] {rel(oof_path, root)} \u2014 \"",
    "                f\"expected companion '{test_name}' not found\"",
    "            )",
    "    return pairs",
    "",
    "",
    "paired_paths = find_prediction_pairs(PREDICTION_ROOT)",
    "",
    "MANIFEST_PATH.write_text(",
    "    json.dumps(paired_paths, indent=2, sort_keys=True),",
    '    encoding="utf-8"',
    ")",
    "",
    'print(f"\\nDiscovered {len(paired_paths)} OOF/test pairs.")',
    'print(f"Manifest saved -> {MANIFEST_PATH}")',
    "print()",
    "for i, (ok, tk) in enumerate(paired_paths.items()):",
    '    print(f"  OOF  : {ok}")',
    '    print(f"  test : {tk}")',
    "    print()",
    "    if i >= 4:",
    '        print(f"  ... ({len(paired_paths) - 5} more pairs)")',
    "        break",
)
cell3["source"] = new_src3
cell3["outputs"] = []
cell3["execution_count"] = None

# ─────────────────────────────────────────────────────────────────────────────
# 2. Step 6d — route ALL artifacts to RUN_DIR
# ─────────────────────────────────────────────────────────────────────────────
# Find cell by scanning for the blend_npy_path line
idx6d = None
for i, c in enumerate(nb["cells"]):
    src_text = "".join(c.get("source", []))
    if "hill_climbing_test_probabilities" in src_text and c["cell_type"] == "code":
        idx6d = i
        break
assert idx6d is not None, "Step 6d cell not found"

new_src6d = src(
    "# \u2500\u2500 Build test-probability blend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "test_blend = np.zeros(n_test, dtype=np.float64)",
    "for name, weight in final_weights.items():",
    "    test_blend += weight * test_dict[name]",
    "",
    "# All outputs go to RUN_DIR (timestamped folder created in Step 3)",
    "",
    "# \u2500\u2500 1. Blended test probabilities (.npy) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    'blend_npy_path = RUN_DIR / "hill_climbing_test_probabilities.npy"',
    "np.save(blend_npy_path, test_blend)",
    'print(f"Saved test probabilities -> {blend_npy_path}")',
    "",
    "# \u2500\u2500 2. Submission CSV \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    'submission_path = RUN_DIR / "hill_climbing_submission.csv"',
    "pd.DataFrame(",
    '    {"id": test_ids, TARGET_COL: test_blend}',
    ").to_csv(submission_path, index=False)",
    'print(f"Saved submission CSV    -> {submission_path}")',
    "",
    "# \u2500\u2500 3. Result report (JSON) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "result = {",
    '    "run_timestamp"      : RUN_TIMESTAMP,',
    '    "run_dir"            : str(RUN_DIR),',
    '    "metric"             : "roc_auc",',
    '    "device"             : str(DEVICE),',
    '    "total_candidates"   : len(ordered_models),',
    '    "selected_model_count": len(selected_models),',
    '    "selected_models"    : selected_models,',
    '    "final_oof_auc"      : best_cv,',
    '    "individual_oof_aucs": {',
    "        k: round(v, 10)",
    "        for k, v in sorted(",
    "            individual_aucs.items(), key=lambda kv: kv[1], reverse=True",
    "        )",
    "    },",
    '    "weights": dict(',
    "        sorted(final_weights.items(), key=lambda kv: kv[1], reverse=True)",
    "    ),",
    "}",
    'result_json_path = RUN_DIR / "hill_climbing_result.json"',
    'result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")',
    'print(f"Saved result report     -> {result_json_path}")',
    "",
    "# \u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    'print("\\n" + "=" * 60)',
    'print(f"Hill-Climbing Run : {RUN_TIMESTAMP}")',
    'print("=" * 60)',
    'print(f"Final OOF AUC     : {best_cv:.8f}")',
    'print(f"Selected models   : {len(selected_models)} / {len(ordered_models)}")',
    'print(f"Output directory  : {RUN_DIR}")',
    "print()",
    'print("-" * 60)',
    'print(f"FINAL WEIGHTS  (normalised, sum = {sum(final_weights.values()):.6f})")',
    'print("-" * 60)',
    "for model, w in sorted(",
    "    final_weights.items(), key=lambda kv: kv[1], reverse=True",
    "):",
    "    bar = '#' * int(w * 40)",
    '    print(f"{w:6.4f}  {bar:<40}  {model}")',
    "",
    'print(f"\\nTest prob range : [{test_blend.min():.6f}, {test_blend.max():.6f}]")',
    'print(f"Test prob mean  : {test_blend.mean():.6f}")',
)
nb["cells"][idx6d]["source"] = new_src6d
nb["cells"][idx6d]["outputs"] = []
nb["cells"][idx6d]["execution_count"] = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP / feature-importance cell (c8d13307)
# ─────────────────────────────────────────────────────────────────────────────
idx_shap, cell_shap = find_cell(nb, "c8d13307")
assert cell_shap is not None, "SHAP cell c8d13307 not found"

new_src_shap = src(
    "# ============================================================",
    "# Top Contributing Models \u2014 Permutation Importance + SHAP",
    "# ============================================================",
    "# Prerequisites (must run Steps 3-6 first):",
    "#   oof_dict, test_dict, individual_aucs, final_weights,",
    "#   ordered_models, best_cv, y, n_train, RUN_DIR",
    "# Optional (Step 7 meta-model):",
    "#   meta_model (trained PyTorch MLP), X_train_meta, model_names",
    "import matplotlib.pyplot as plt",
    "",
    "# \u2500\u2500 Helper: exact tie-aware ROC-AUC \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "def _roc_auc(y_true, scores):",
    "    y_true = np.asarray(y_true, dtype=np.int8)",
    "    scores = np.asarray(scores, dtype=np.float64)",
    "    pos = int(y_true.sum()); neg = len(y_true) - pos",
    "    order  = np.argsort(scores, kind='mergesort')",
    "    ss     = scores[order]",
    "    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]",
    "    ends   = np.r_[starts[1:], len(scores)]",
    "    sr     = np.repeat((starts + 1 + ends) / 2.0, ends - starts)",
    "    ranks  = np.empty(len(scores), dtype=np.float64)",
    "    ranks[order] = sr",
    "    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))",
    "",
    "# \u2500\u2500 1. Permutation Importance on OOF predictions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# For each selected model: replace its OOF contribution with a shuffled",
    "# version and measure the drop in ensemble OOF AUC.",
    "print('Computing permutation importance ...')",
    "rng = np.random.default_rng(42)",
    "",
    "# Reconstruct the weighted OOF blend",
    "oof_blend_base = np.zeros(n_train, dtype=np.float64)",
    "for name, w in final_weights.items():",
    "    oof_blend_base += w * oof_dict[name]",
    "baseline_auc = _roc_auc(y, oof_blend_base)",
    "",
    "perm_importance = {}",
    "for name, w in final_weights.items():",
    "    shuffled  = rng.permutation(oof_dict[name])",
    "    perturbed = oof_blend_base - w * oof_dict[name] + w * shuffled",
    "    drop = baseline_auc - _roc_auc(y, perturbed)",
    "    perm_importance[name] = max(drop, 0.0)   # clip noise negatives",
    "",
    "# Individual OOF AUC for ALL valid models",
    "indiv_auc_imp = {name: individual_aucs[name] for name in oof_dict}",
    "",
    'print(f"Baseline blend OOF AUC : {baseline_auc:.8f}")',
    'print(f"Models with non-zero perm importance: {sum(v > 0 for v in perm_importance.values())}")',
    "",
    "# \u2500\u2500 2. SHAP \u2014 KernelExplainer on the trained meta MLP \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# Only runs if meta_model and X_train_meta exist (Step 7 completed).",
    "shap_imp = None",
    "try:",
    "    import shap",
    "    meta_model.eval()",
    "    def _predict_proba(X_np):",
    "        with torch.no_grad():",
    "            t   = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)",
    "            out = torch.softmax(meta_model(t), dim=-1).cpu().numpy()",
    "        return out",
    "    bg_size    = min(100, len(X_train_meta))",
    "    bg_idx     = rng.choice(len(X_train_meta), bg_size, replace=False)",
    "    background = X_train_meta[bg_idx].astype(np.float32)",
    "    ex_idx     = rng.choice(len(X_train_meta), min(300, len(X_train_meta)), replace=False)",
    "    X_explain  = X_train_meta[ex_idx].astype(np.float32)",
    "    print('Running SHAP KernelExplainer (may take a few minutes) ...')",
    "    explainer   = shap.KernelExplainer(_predict_proba, background)",
    "    shap_values = explainer.shap_values(X_explain, silent=True)",
    "    # shap_values: list[num_classes] each (n_explain, n_features)",
    "    # Use positive-class SHAP values for ranking",
    "    sv_pos = shap_values[1] if isinstance(shap_values, list) else shap_values",
    "    sv_mean = np.abs(sv_pos).mean(axis=0)   # (n_features,)",
    "    shap_imp = {model_names[i]: float(sv_mean[i]) for i in range(len(model_names))}",
    "    print('SHAP computation complete.')",
    "except Exception as e:",
    '    print(f"[INFO] SHAP skipped: {e}")',
    "",
    "# \u2500\u2500 3. Plot: Top-10 models (dark theme) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "TOP_N  = 10",
    "plt.style.use('dark_background')",
    "",
    "def top_n(imp_dict, n):",
    "    items  = sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)[:n]",
    "    labels = [k.split('/')[-1].replace('_oof.npy','').replace('oof_','') for k,_ in items]",
    "    vals   = [v for _, v in items]",
    "    return labels, vals",
    "",
    "def _bar(ax, labels, vals, title, color, xlabel):",
    "    ax.set_facecolor('#0d1117')",
    "    bars = ax.barh(range(len(labels)), vals, color=color,",
    "                   edgecolor='#30363d', linewidth=0.5)",
    "    ax.set_yticks(range(len(labels)))",
    "    ax.set_yticklabels(labels, fontsize=9, color='#e6edf3')",
    "    ax.invert_yaxis()",
    "    ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold', pad=8)",
    "    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)",
    "    ax.tick_params(colors='#8b949e', labelsize=8)",
    "    for spine in ax.spines.values():",
    "        spine.set_edgecolor('#30363d')",
    "    for bar, val in zip(bars, vals):",
    "        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,",
    "                f'{val:.5f}', va='center', ha='left', fontsize=7.5, color='#e6edf3')",
    "",
    "n_plots = 3 if shap_imp else 2",
    "fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 7))",
    "fig.patch.set_facecolor('#161b22')",
    "if n_plots == 2:",
    "    axes = list(axes)",
    "",
    "perm_labels, perm_vals = top_n(perm_importance, TOP_N)",
    "_bar(axes[0], perm_labels, perm_vals,",
    "     f'Top {TOP_N} \u2014 Permutation Importance\\n(AUC drop when model shuffled)',",
    "     '#f85149', 'AUC drop')",
    "",
    "auc_labels, auc_vals = top_n(indiv_auc_imp, TOP_N)",
    "_bar(axes[1], auc_labels, auc_vals,",
    "     f'Top {TOP_N} \u2014 Individual OOF AUC',",
    "     '#388bfd', 'OOF AUC')",
    "",
    "if shap_imp:",
    "    shap_labels, shap_vals = top_n(shap_imp, TOP_N)",
    "    _bar(axes[2], shap_labels, shap_vals,",
    "         f'Top {TOP_N} \u2014 SHAP Mean |Value|\\n(positive class, meta-MLP)',",
    "         '#bc8cff', 'Mean |SHAP|')",
    "",
    "fig.suptitle('Top Contributing Base Models', color='#e6edf3',",
    "             fontsize=14, fontweight='bold', y=1.01)",
    "plt.tight_layout()",
    "",
    "plot_path = RUN_DIR / 'top_model_importance.png'",
    "fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())",
    "plt.show()",
    'print(f"Plot saved -> {plot_path}")',
    "",
    "# \u2500\u2500 4. Print ranked tables \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "print('\\n--- Permutation Importance (selected models, ranked) ---')",
    "for rank, (name, val) in enumerate(",
    "    sorted(perm_importance.items(), key=lambda kv: kv[1], reverse=True), 1",
    "):",
    "    sel_w = final_weights.get(name, 0)",
    '    print(f"  {rank:>3}. drop={val:.6f}  w={sel_w:.4f}  {name}")',
    "",
    "print('\\n--- Individual OOF AUC (top 15, all valid models) ---')",
    "for rank, (name, val) in enumerate(",
    "    sorted(indiv_auc_imp.items(), key=lambda kv: kv[1], reverse=True)[:15], 1",
    "):",
    "    tag = ' [selected]' if name in final_weights else ''",
    '    print(f"  {rank:>3}. {val:.8f}  {name}{tag}")',
    "",
    "if shap_imp:",
    "    print('\\n--- SHAP Mean |Value| (top 15, positive class) ---')",
    "    for rank, (name, val) in enumerate(",
    "        sorted(shap_imp.items(), key=lambda kv: kv[1], reverse=True)[:15], 1",
    "    ):",
    '        print(f"  {rank:>3}. {val:.6f}  {name}")',
)
cell_shap["source"] = new_src_shap
cell_shap["outputs"] = []
cell_shap["execution_count"] = None

# ─────────────────────────────────────────────────────────────────────────────
# Write patched notebook
# ─────────────────────────────────────────────────────────────────────────────
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched {NB_PATH}  ({NB_PATH.stat().st_size:,} bytes)")
print(f"  cell_step3  idx={idx3}")
print(f"  step6d      idx={idx6d}")
print(f"  shap cell   idx={idx_shap}")
