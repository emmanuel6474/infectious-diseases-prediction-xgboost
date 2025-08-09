
import json, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm.auto import tqdm

INPUT_PATH = "infectious-diseases-by-county-year-and-sex.csv"
OUT_PREFIX = "infectious_diseases_xgb_temporal_noleak"
RANDOM_STATE = 42
PRED_THRESHOLD = 0.80
TARGET_QUANTILE = 0.85
N_ESTIMATORS = 500
MAX_DEPTH = 4
ETA = 0.08

def confusion_matrix_2x2(y_true, y_pred):
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tp = int(((y_true==1)&(y_pred==1)).sum())
    return np.array([[tn, fp],[fn, tp]])

def roc_curve_np(y_true, scores):
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P = tp[-1] if tp.size>0 else 0
    N = fp[-1] if fp.size>0 else 0
    tpr = tp / np.clip(P, 1, None)
    fpr = fp / np.clip(N, 1, None)
    return fpr, tpr

def auc_trapz(x, y):
    if len(x) < 2: return float("nan")
    o = np.argsort(x)
    return float(np.trapz(y[o], x[o]))

def pick_region_col(df):
    cols = {c: c.lower() for c in df.columns}
    for key in ["state","province","region","admin","county"]:
        for c,cl in cols.items():
            if key in cl:
                return c
    return "County" if "County" in df.columns else list(df.columns)[0]

steps = ["load","build_target","features_encode","temporal_split","train","predict","metrics","plots_save","region_eval","save_artifacts"]
times = {}
metrics_table = None

for step in tqdm(steps, desc="pipeline", leave=True):
    t0 = time.perf_counter()

    if step == "load":
        df = pd.read_csv(INPUT_PATH)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

    elif step == "build_target":
        req = ["County","Disease","Sex","Year","Population","Cases"]
        missing = [c for c in req if c not in df.columns]
        if missing: raise ValueError(f"Missing required columns: {missing}")
        df = df.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
        df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce").fillna(0)
        df = df.dropna(subset=["Year","Population"]).reset_index(drop=True)
        df = df.sort_values(["County","Disease","Sex","Year"])
        df["Cases_next"] = df.groupby(["County","Disease","Sex"])["Cases"].shift(-1)
        thr = df["Cases_next"].quantile(TARGET_QUANTILE)
        df["target"] = (df["Cases_next"] >= thr).astype(int)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

    elif step == "features_encode":
        leak_cols = ["Cases","Rate","Lower_95__CI","Upper_95__CI","Cases_next","target"]
        keep_cols = [c for c in df.columns if c not in leak_cols]
        X = df[keep_cols].copy()
        y = df["target"].astype(int).to_numpy()
        cat_cols = [c for c in X.columns if X[c].dtype == object]
        for c in cat_cols:
            X[c], _ = pd.factorize(X[c], sort=True)
        feat_names = X.columns.tolist()
        region_col = pick_region_col(df)
        regions_all = df[region_col].values if not pd.api.types.is_numeric_dtype(df[region_col]) else df[region_col].astype(int).values

    elif step == "temporal_split":
        years = np.sort(X["Year"].unique())
        if len(years) < 3:
            split_year = years[-1]
        else:
            split_year = years[int(len(years)*0.8)]
        tr_mask = X["Year"] <= split_year
        te_mask = X["Year"] >  split_year
        if te_mask.sum() == 0:
            te_mask = X["Year"] == years[-1]
            tr_mask = ~te_mask
        X_train, X_test = X[tr_mask].reset_index(drop=True), X[te_mask].reset_index(drop=True)
        y_train, y_test = y[tr_mask.values], y[te_mask.values]
        regions_test = regions_all[te_mask.values]
        dtrain = xgb.DMatrix(X_train.to_numpy(float), label=y_train, feature_names=feat_names)
        dtest  = xgb.DMatrix(X_test.to_numpy(float),  label=y_test,  feature_names=feat_names)
        pos = float(y_train.sum()); neg = float(len(y_train) - y_train.sum())
        spw = max(neg / max(pos,1.0), 1.0)
        params = {
            "objective": "binary:logistic",
            "max_depth": MAX_DEPTH,
            "eta": ETA,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "auc",
            "seed": RANDOM_STATE,
            "nthread": -1,
            "scale_pos_weight": spw,
        }

    elif step == "train":
        iter_times = []
        bst = None
        pbar = tqdm(total=N_ESTIMATORS, desc="xgb-train", leave=True)
        for _ in range(N_ESTIMATORS):
            tt = time.perf_counter()
            bst = xgb.train(params, dtrain, num_boost_round=1, xgb_model=bst, verbose_eval=False)
            iter_times.append(time.perf_counter() - tt)
            pbar.update(1)
        pbar.close()

    elif step == "predict":
        t_pred0 = time.perf_counter()
        proba_test = bst.predict(dtest)
        t_pred1 = time.perf_counter()

    elif step == "metrics":
        fpr, tpr = roc_curve_np(y_test, proba_test)
        auc = auc_trapz(fpr, tpr)
        thr_grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
        rows = []
        for thr in thr_grid:
            pred = (proba_test >= thr).astype(int)
            cm = confusion_matrix_2x2(y_test, pred)
            tn, fp = cm[0]; fn, tp = cm[1]
            acc = (tn+tp)/np.clip(cm.sum(),1,None)
            prec = tp/np.clip(tp+fp,1,None) if (tp+fp)>0 else 0.0
            rec  = tp/np.clip(tp+fn,1,None) if (tp+fn)>0 else 0.0
            f1   = (2*prec*rec)/np.clip(prec+rec,1e-9,None) if (prec+rec)>0 else 0.0
            tnr  = tn/np.clip(tn+fp,1,None)
            bacc = 0.5*(tnr+rec)
            ber  = 1.0 - bacc
            rows.append([thr, auc, acc, bacc, ber, prec, rec, f1, int(tn), int(fp), int(fn), int(tp)])
        metrics_table = pd.DataFrame(rows, columns=["threshold","AUC","Accuracy","BalancedAccuracy","BalancedErrorRate","Precision","Recall","F1","TN","FP","FN","TP"])
        print("\\n=== Metrics by threshold (temporal test) ===")
        print(metrics_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    elif step == "plots_save":
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC (temporal test)")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUT_PREFIX}_roc.png", dpi=300)
        try:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(X_test.to_numpy(float))
            shap.summary_plot(shap_values, features=X_test, feature_names=feat_names, show=False)
            plt.tight_layout(); plt.savefig(f"{OUT_PREFIX}_beeswarm.png", dpi=220)
        except Exception:
            shap_contribs = bst.predict(dtest, pred_contribs=True)
            sv = shap_contribs[:, :-1]
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(-mean_abs)
            k = min(20, sv.shape[1]); idx = order[:k]
            names = [feat_names[i] for i in idx]
            plt.figure(figsize=(8,10))
            for rank, j in enumerate(idx[::-1]):
                vals = sv[:, j]
                yy = np.full_like(vals, rank, dtype=float) + (np.random.rand(len(vals))-0.5)*0.6
                plt.scatter(vals, yy, s=8, alpha=0.7)
            plt.plot(mean_abs[idx][::-1], np.arange(k), linestyle="--")
            plt.yticks(np.arange(k), [names[::-1][t] for t in range(k)])
            plt.xlabel("SHAP value (log-odds)"); plt.title("XGBoost SHAP Beeswarm (Top Features)")
            plt.tight_layout(); plt.savefig(f"{OUT_PREFIX}_beeswarm.png", dpi=200)

    elif step == "region_eval":
        reg_col_name = pick_region_col(df)
        reg_vals = pd.Series(regions_test).astype(str)
        reg_unique = reg_vals.unique().tolist()
        rows_reg = []
        for r in reg_unique:
            m = (reg_vals == r).values
            if m.sum() < 20:
                continue
            y_r = y_test[m]
            p_r = proba_test[m]
            if len(np.unique(y_r)) < 2:
                continue
            fpr_r, tpr_r = roc_curve_np(y_r, p_r)
            auc_r = auc_trapz(fpr_r, tpr_r)
            pred_r = (p_r >= PRED_THRESHOLD).astype(int)
            cm = confusion_matrix_2x2(y_r, pred_r)
            tn, fp = cm[0]; fn, tp = cm[1]
            acc = (tn+tp)/np.clip(cm.sum(),1,None)
            prec = tp/np.clip(tp+fp,1,None) if (tp+fp)>0 else 0.0
            rec  = tp/np.clip(tp+fn,1,None) if (tp+fn)>0 else 0.0
            f1   = (2*prec*rec)/np.clip(prec+rec,1e-9,None) if (prec+rec)>0 else 0.0
            tnr  = tn/np.clip(tn+fp,1,None)
            bacc = 0.5*(tnr+rec)
            ber  = 1.0 - bacc
            rows_reg.append([r, auc_r, acc, bacc, ber, prec, rec, f1, int(tn), int(fp), int(fn), int(tp), int(m.sum())])
        region_table = pd.DataFrame(rows_reg, columns=["region","AUC","Accuracy","BalancedAccuracy","BalancedErrorRate","Precision","Recall","F1","TN","FP","FN","TP","n_test"])
        region_table = region_table.sort_values("AUC", ascending=False).reset_index(drop=True)
        print(f"\\n=== Region-wise evaluation at threshold {PRED_THRESHOLD:.2f} (temporal test) ===")
        if not region_table.empty:
            print(region_table.to_string(index=False, float_format=lambda x: f\"{x:.4f}\"))
        else:
            print("No region with sufficient size and both classes in test.")
        region_table.to_csv(f"{OUT_PREFIX}_region_eval_thr{int(PRED_THRESHOLD*100):02d}.csv", index=False)

    elif step == "save_artifacts":
        metrics_table.to_csv(f"{OUT_PREFIX}_metrics_by_threshold.csv", index=False)
        with open(f"{OUT_PREFIX}_metrics_by_threshold.json","w") as f:
            json.dump({"thresholds": metrics_table.to_dict(orient="records"),
                       "target_quantile": TARGET_QUANTILE}, f, indent=2)
        it = np.array(iter_times, dtype=float)
        print("\\nXGB timing:")
        print(f"- iterations: {len(it)}  total: {it.sum():.2f}s  avg/iter: {it.mean()*1000:.1f} ms")
        print(f"- test predict time: {(t_pred1 - t_pred0):.3f}s")
        print("Artifacts saved:",
              f"{OUT_PREFIX}_roc.png, {OUT_PREFIX}_beeswarm.png, {OUT_PREFIX}_metrics_by_threshold.csv/.json, "
              f"{OUT_PREFIX}_region_eval_thr{int(PRED_THRESHOLD*100):02d}.csv")

    times[step] = time.perf_counter() - t0

print("\\nPipeline timing (s):")
for k,v in times.items():
    print(f"- {k}: {v:.2f}")
