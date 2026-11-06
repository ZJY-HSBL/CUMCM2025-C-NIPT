import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

CN = {
    "id": "孕妇代码",
    "ga": "检测孕周",
    "bmi": "孕妇BMI",
    "y": "Y染色体浓度",
    "age": "年龄",
    "height": "身高",
    "weight": "体重",
    "gc_total": "GC含量",
    "gc13": "13号染色体的GC含量",
    "gc18": "18号染色体的GC含量",
    "gc21": "21号染色体的GC含量",
    "reads_total": "原始读段数",
    "reads_unique": "唯一比对的读段数  ",
    "map_ratio": "在参考基因组上比对的比例",
    "dup_ratio": "重复读段的比例",
    "filter_ratio": "被过滤掉读段数的比例",
}

CONFIG = {
    "data_path": "附件.xlsx",
    "sheet_name": "男胎检测数据",
    "kmin": 2,
    "kmax": 4,
    "tau": 0.90,
    "bootstrap": 500,
    "outdir": "outputs_q3_surv",
    "cox_penalizer": 0.1,
    "cox_max_iter": 1000,
    "cox_tolerance": 1e-6,
    "random_state": 42
}

def parse_ga_to_weeks(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace('周', 'w')
    try:
        if 'w' in s and '+' in s:
            w, d = s.split('+', 1)
            w = w.replace('w', '')
            return float(w) + float(d) / 7.0
        if '+' in s:
            w, d = s.split('+', 1)
            return float(w) + float(d) / 7.0
        if 'w' in s:
            return float(s.replace('w', ''))
        return float(s)
    except:
        return np.nan

def ward_seed_bgm_strata(bmi: np.ndarray) -> np.ndarray:
    X = bmi.reshape(-1, 1)
    best_score = -1
    best_labels = None
    best_k = CONFIG["kmin"]

    for k in range(CONFIG["kmin"], CONFIG["kmax"] + 1):
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ward.fit_predict(X)
        sil_score = silhouette_score(X, labels) if k > 1 else -1
        ch_score = calinski_harabasz_score(X, labels) if k > 1 else -1
        weighted_score = 0.7 * sil_score + 0.3 * (ch_score / 1000.0)

        if weighted_score > best_score:
            best_score = weighted_score
            best_labels = labels
            best_k = k

    means_init = np.array([X[best_labels == j].mean(axis=0) for j in range(best_k)])
    weights_init = np.array([(best_labels == j).mean() for j in range(best_k)])

    bgm = BayesianGaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=CONFIG["random_state"],
        max_iter=1000,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=10.0,
        init_params="random"
    )
    bgm.means_init = means_init
    bgm.weights_init = weights_init
    bgm.fit(X)
    gmm_labels = bgm.predict(X)

    bmi_means = [(j, bmi[gmm_labels == j].mean()) for j in range(best_k)]
    sorted_means = sorted(bmi_means, key=lambda x: x[1])
    label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_means)}
    sorted_labels = np.vectorize(label_mapping.get)(gmm_labels)

    return sorted_labels

def build_subject_cov(df_rows: pd.DataFrame) -> pd.DataFrame:
    subject_cov = df_rows.groupby("ID").agg({
        "BMI": "median", "Age": "median", "Height": "median", "Weight": "median",
        "GC": "median", "GC13": "median", "GC18": "median", "GC21": "median",
        "Reads": "median", "ReadsU": "median", "MapR": "median", "DupR": "median", "FiltR": "median",
        "BMI_layer": "first"
    }).reset_index()
    return subject_cov

def construct_time_interval(rows_one: pd.DataFrame, thr=0.04) -> Tuple[float, float]:
    sorted_records = rows_one.sort_values("GA").loc[:, ["GA", "Y"]].dropna()
    if sorted_records.empty:
        return (np.nan, np.nan)

    达标索引 = sorted_records.index[sorted_records["Y"] >= thr].tolist()
    if 达标索引:
        首次达标索引 = 达标索引[0]
        首次达标孕周 = float(sorted_records.loc[首次达标索引, "GA"])
        达标前记录 = sorted_records[sorted_records["GA"] < 首次达标孕周]
        达标前记录 = 达标前记录[达标前记录["Y"] < thr]

        if len(达标前记录) > 0:
            最后未达标孕周 = float(达标前记录["GA"].max())
            return (最后未达标孕周, 首次达标孕周)
        else:
            return (10.0, 首次达标孕周)
    else:
        最后检测孕周 = float(sorted_records["GA"].max())
        return (最后检测孕周, np.inf)

def build_survival_table(df_rows: pd.DataFrame) -> pd.DataFrame:
    survival_records = []
    for 孕妇ID, 孕妇记录 in df_rows.groupby("ID"):
        下界, 上界 = construct_time_interval(孕妇记录)
        if np.isnan(下界) and np.isnan(上界):
            continue

        if np.isfinite(上界):
            if 下界 > 10.0:
                事件时间 = (下界 + 上界) / 2.0
            else:
                事件时间 = 上界
            事件标记 = 1
            事件时间 = max(10.0, min(25.0, float(事件时间)))
        else:
            事件时间 = 25.0
            事件标记 = 0

        survival_records.append({
            "ID": 孕妇ID,
            "T": 事件时间,
            "E": 事件标记,
            "lower": 下界,
            "upper": 上界
        })
    return pd.DataFrame(survival_records)

def predict_group_cdf(model, X_subjects: pd.DataFrame, times: np.ndarray, model_type="cox") -> np.ndarray:
    if model_type == "cox":
        生存函数 = model.predict_survival_function(X_subjects, times=times)
    else:
        生存函数 = model.predict_survival_function(X_subjects, times=times)

    平均生存函数 = 生存函数.mean(axis=1).values
    群体CDF = 1.0 - 平均生存函数
    return 群体CDF

def earliest_ga_above_tau(cdf: np.ndarray, times: np.ndarray, tau: float) -> float:
    达标索引 = np.where(cdf >= tau)[0]
    if len(达标索引) > 0:
        return float(times[达标索引[0]])
    else:
        return np.nan

def bootstrap_ga_star(
        df_rows_layer: pd.DataFrame,
        subj_cov_layer: pd.DataFrame,
        features : List[str],
        config=CONFIG
) -> Tuple[List[float], List[Dict]]:
    rng = np.random.default_rng(config["random_state"])
    孕妇ID列表 = subj_cov_layer["ID"].values
    孕周网格 = np.round(np.arange(10.0, 25.0 + 1e-9, 0.1), 1)

    ga_star_list = []
    feature_effects = []

    for bootstrap_idx in range(config["bootstrap"]):
        if (bootstrap_idx + 1) % 20 == 0:
            print(f"  Bootstrap抽样进度：{bootstrap_idx + 1}/{config['bootstrap']}次")

        抽样索引 = rng.integers(0, len(孕妇ID列表), size=len(孕妇ID列表))
        抽样ID集合 = set(孕妇ID列表[抽样索引])
        抽样数据 = df_rows_layer[df_rows_layer["ID"].isin(抽样ID集合)].copy()

        if 抽样数据.empty:
            ga_star_list.append(np.nan)
            feature_effects.append({})
            continue

        近阈值掩码 = 抽样数据["Y"].between(0.035, 0.045, inclusive="both")
        if 近阈值掩码.sum() > 0:
            扰动概率 = 抽样数据.loc[近阈值掩码, "Y"].apply(
                lambda v: float(np.clip(np.exp(-((v - 0.04) ** 2) / (2 * (0.0025 ** 2))) * 0.3, 0, 1))
            )
            随机数 = rng.uniform(0, 1, size=近阈值掩码.sum())
            扰动索引 = 抽样数据.loc[近阈值掩码].index

            for idx, (p, u) in zip(扰动索引, zip(扰动概率.values, 随机数)):
                if u < p:
                    if 抽样数据.loc[idx, "Y"] >= 0.04:
                        抽样数据.loc[idx, "Y"] = 0.0349
                    else:
                        抽样数据.loc[idx, "Y"] = 0.0401

        抽样生存表 = build_survival_table(抽样数据)

        if 抽样生存表.empty:
            ga_star_list.append(np.nan)
            feature_effects.append({})
            continue

        抽样个体特征 = subj_cov_layer[subj_cov_layer["ID"].isin(抽样生存表["ID"])].copy()
        模型数据 = pd.merge(抽样生存表, 抽样个体特征, on="ID", how="inner")

        if 模型数据["E"].sum() == 0:
            ga_star_list.append(np.nan)
            feature_effects.append({})
            continue

        有效特征 = [f for f in features if f in 模型数据.columns]
        if not 有效特征:
            ga_star_list.append(np.nan)
            feature_effects.append({})
            continue

        拟合数据 = 模型数据[["T", "E"] + 有效特征].copy()

        cox_model = CoxPHFitter()
        cox_model.max_iter = config["cox_max_iter"]
        cox_model.tolerance = config["cox_tolerance"]
        cox_model.penalizer = config["cox_penalizer"]
        cox_model.show_progress=False

        try:
            cox_model.fit(拟合数据, duration_col="T", event_col="E")
        except Exception as e:
            print(f"  第{bootstrap_idx + 1}次抽样模型拟合失败：{str(e)[:50]}...")
            ga_star_list.append(np.nan)
            feature_effects.append({})
            continue

        summary = cox_model.summary.copy()
        summary = summary.reset_index().rename(columns={"index": "covariate"})

        if "hazard_ratio" in summary.columns:
            hr_col = "hazard_ratio"
            lower_col = "lower_95%"
            upper_col = "upper_95%"
        else:
            hr_col = "exp(coef)"
            lower_col = "exp(coef) lower 95%"
            upper_col = "exp(coef) upper 95%"

        hr_dict = {}
        for _, row in summary.iterrows():
            feat = row["covariate"]
            if feat in 有效特征:
                hr_dict[feat] = {
                    "hr": float(row[hr_col]),
                    "lower_95ci": float(row[lower_col]),
                    "upper_95ci": float(row[upper_col])
                }
        feature_effects.append(hr_dict)

        群体生存函数 = cox_model.predict_survival_function(抽样个体特征[有效特征], times=孕周网格)
        群体达标率 = 1.0 - 群体生存函数.mean(axis=1).values

        达标索引 = np.where(群体达标率 >= config["tau"])[0]
        if len(达标索引) > 0:
            ga_star = float(孕周网格[达标索引[0]])
        else:
            ga_star = np.nan

        ga_star_list.append(ga_star)

    ga_star_clean = [g for g in ga_star_list if not np.isnan(g)]
    feature_effects_clean = [eff for eff, g in zip(feature_effects, ga_star_list) if not np.isnan(g)]

    print(f"Bootstrap抽样完成：共{config['bootstrap']}次，有效结果{len(ga_star_clean)}次")
    return ga_star_clean, feature_effects_clean

def fit_cox_model(拟合数据, config):
    cox_model = CoxPHFitter()
    cox_model.max_iter = config["cox_max_iter"]
    cox_model.tolerance = config["cox_tolerance"]
    cox_model.penalizer = config["cox_penalizer"]

    try:
        cox_model.fit(
            拟合数据,
            duration_col="T",
            event_col="E",
            show_progress=False
        )
        print("Cox模型拟合成功")
        return cox_model
    except Exception as e:
        print(f"Cox模型拟合失败：{str(e)}")
        return None

def calculate_c_index(cox_model, 模型数据):
    c_index = concordance_index(
        模型数据["T"],
        -cox_model.predict_partial_hazard(模型数据),
        模型数据["E"]
    )
    return c_index

def plot_and_save_curve(群体CDF, ga_star, layer, 孕周网格, tau, outdir):
    plt.figure(figsize=(8, 5))

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'p']

    style_idx = layer % len(line_styles)
    marker_idx = layer % len(markers)

    plt.plot(
        孕周网格, 群体CDF,
        color='black',
        linestyle=line_styles[style_idx],
        marker=markers[marker_idx],
        markersize=3,
        markevery=20,
        label=f"分层 {layer} CDF"
    )

    plt.axhline(
        y=tau,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=f"目标阈值τ={tau:.2f}"
    )

    if not np.isnan(ga_star):
        plt.axvline(
            x=ga_star,
            linestyle="-.",
            color="black",
            linewidth=1.0,
            label=f"GA*: {ga_star:.1f}周"
        )

    plt.xlabel("孕周（周）", fontsize=15)
    plt.ylabel("达到Y≥4%的比例", fontsize=15)
    plt.legend(fontsize=12)

    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False

    plt.tight_layout()
    plt.show()

def save_results(结果汇总, outdir):
    结果表 = pd.DataFrame(结果汇总).sort_values("BMI分层")
    结果表.to_csv(os.path.join(outdir, "各分层结果汇总.csv"), index=False)

def main():
    try:
        df_raw = pd.read_excel(CONFIG["data_path"], sheet_name=CONFIG["sheet_name"])
        print(f"成功读取数据：{CONFIG['data_path']} (工作表：{CONFIG['sheet_name']})")
    except Exception as e:
        print(f"读取数据失败：{str(e)}")
        print("请检查CONFIG中的data_path和sheet_name是否正确")
        return

    df = pd.DataFrame({
        "ID": df_raw[CN["id"]],
        "GA": df_raw[CN["ga"]].apply(parse_ga_to_weeks),
        "BMI": pd.to_numeric(df_raw[CN["bmi"]], errors="coerce"),
        "Y": pd.to_numeric(df_raw[CN["y"]], errors="coerce"),
        "Age": pd.to_numeric(df_raw.get(CN["age"]), errors="coerce"),
        "Height": pd.to_numeric(df_raw.get(CN["height"]), errors="coerce"),
        "Weight": pd.to_numeric(df_raw.get(CN["weight"]), errors="coerce"),
        "GC": pd.to_numeric(df_raw.get("GC含量", df_raw.get(CN["gc_total"])), errors="coerce"),
        "GC13": pd.to_numeric(df_raw.get(CN["gc13"]), errors="coerce"),
        "GC18": pd.to_numeric(df_raw.get(CN["gc18"]), errors="coerce"),
        "GC21": pd.to_numeric(df_raw.get(CN["gc21"]), errors="coerce"),
        "Reads": pd.to_numeric(df_raw.get(CN["reads_total"]), errors="coerce"),
        "ReadsU": pd.to_numeric(df_raw.get(CN["reads_unique"]), errors="coerce"),
        "MapR": pd.to_numeric(df_raw.get(CN["map_ratio"]), errors="coerce"),
        "DupR": pd.to_numeric(df_raw.get(CN["dup_ratio"]), errors="coerce"),
        "FiltR": pd.to_numeric(df_raw.get(CN["filter_ratio"]), errors="coerce"),
    })

    df = df.dropna(subset=["ID", "GA", "BMI", "Y"])
    df = df[(df["GA"] >= 10) & (df["GA"] <= 25)]
    print(f"数据预处理完成：保留有效记录 {df.shape[0]} 条，涉及孕妇 {df['ID'].nunique()} 人")

    孕妇BMI = df.groupby("ID")["BMI"].median().reset_index()
    孕妇BMI["BMI_layer"] = ward_seed_bgm_strata(孕妇BMI["BMI"].values)
    df = df.merge(孕妇BMI[["ID", "BMI_layer"]], on="ID", how="left")
    分层数量 = df["BMI_layer"].nunique()
    print(f"BMI分层完成：共分为 {分层数量} 层")

    个体特征 = build_subject_cov(df)

    feature_cols = ["BMI", "Age", "Height", "Weight", "GC", "GC13", "GC18",
                    "GC21", "Reads", "ReadsU", "MapR", "DupR", "FiltR"]
    feature_variances = 个体特征[feature_cols].var()
    print("\n各特征方差：")
    print(feature_variances.round(6))

    var_threshold = 1e-4
    low_var_features = [feat for feat in feature_cols if feature_variances[feat] < var_threshold]
    high_var_features = [feat for feat in feature_cols if feature_variances[feat] >= var_threshold]

    if low_var_features:
        print(f"\n删除低方差特征：{low_var_features}")
        个体特征 = 个体特征.drop(columns=low_var_features)

        numeric_features = 个体特征.select_dtypes(include=['number'])
        corr_matrix = numeric_features.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='gray',
            fmt='.2f',
            linewidths=0.5
        )

        plt.tight_layout()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

    else:
        print("\n无低方差特征，保留所有特征")

    all_features = high_var_features

    孕周网格 = np.round(np.arange(10.0, 25.0 + 1e-9, 0.1), 1)
    结果汇总 = []

    for layer in sorted(个体特征["BMI_layer"].unique()):
        print(f"\n===== 分析BMI分层 {layer} =====")
        分层记录 = df[df["BMI_layer"] == layer].copy()
        分层个体 = 个体特征[个体特征["BMI_layer"] == layer].copy()
        n_subjects = 分层个体.shape[0]
        print(f"分层 {layer} 包含 {n_subjects} 名孕妇")

        生存表 = build_survival_table(分层记录)
        模型数据 = pd.merge(生存表, 分层个体, on="ID", how="inner")

        if 模型数据.shape[0] < 20:
            print(f"警告：分层 {layer} 样本量过小，跳过分析")
            continue
        if 模型数据["E"].sum() == 0:
            print(f"警告：分层 {layer} 无达标事件，跳过分析")
            continue

        拟合数据 = 模型数据[["T", "E"] + all_features].copy()

        cox_model = fit_cox_model(拟合数据, CONFIG)
        if cox_model is None:
            continue

        c_index = calculate_c_index(cox_model, 模型数据)
        print(f"分层 {layer} 模型C-index：{c_index:.3f}（越接近1越好）")

        X_sub = 分层个体[all_features].copy()
        群体CDF = predict_group_cdf(cox_model, X_sub, 孕周网格, model_type="cox")
        ga_star = earliest_ga_above_tau(群体CDF, 孕周网格, CONFIG["tau"])

        print(f"分层 {layer} 正在进行Bootstrap抽样...")
        ga_stars, feature_effects = bootstrap_ga_star(分层记录, 分层个体, all_features, CONFIG)

        ga_stars_clean = [g for g in ga_stars if not np.isnan(g)]
        if len(ga_stars_clean) > 0:
            ci_low, ci_high = np.percentile(ga_stars_clean, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan

        pad = 0.5
        rec_low = max(10.0, ci_low - pad if not np.isnan(ci_low) else (ga_star - pad if not np.isnan(ga_star) else np.nan))
        rec_high = min(25.0, ci_high + pad if not np.isnan(ci_high) else (ga_star + pad if not np.isnan(ga_star) else np.nan))

        plot_and_save_curve(群体CDF, ga_star, layer, 孕周网格, CONFIG["tau"], CONFIG["outdir"])

        bmi_min, bmi_max, bmi_mean = (
            float(分层个体["BMI"].min()),
            float(分层个体["BMI"].max()),
            float(分层个体["BMI"].mean())
        )
        结果汇总.append({
            "BMI分层": layer,
            "孕妇数量": 分层个体.shape[0],
            "BMI最小值": round(bmi_min, 2),
            "BMI最大值": round(bmi_max, 2),
            "BMI平均值": round(bmi_mean, 2),
            "GA*（周）": None if np.isnan(ga_star) else round(ga_star, 2),
            "95%CI下限": None if np.isnan(ci_low) else round(ci_low, 2),
            "95%CI上限": None if np.isnan(ci_high) else round(ci_high, 2),
            "推荐检测窗口下限": None if np.isnan(rec_low) else round(rec_low, 2),
            "推荐检测窗口上限": None if np.isnan(rec_high) else round(rec_high, 2),
        })

    结果表 = pd.DataFrame(结果汇总).sort_values("BMI分层")
    save_results(结果表, CONFIG["outdir"])

    plt.figure(figsize=(9, 6))

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'p', '*', 'h']

    for i, layer in enumerate(sorted(结果表["BMI分层"].unique())):
        曲线路径 = os.path.join(CONFIG["outdir"], f"rate_curve_layer{layer}.csv")
        if os.path.exists(曲线路径):
            曲线数据 = pd.read_csv(曲线路径)

            style_idx = i % len(line_styles)
            marker_idx = i % len(markers)

            plt.plot(
                曲线数据["GA"],
                曲线数据["Rate"],
                color='black',
                linestyle=line_styles[style_idx],
                marker=markers[marker_idx],
                markersize=4,
                markevery=30,
                label=f"分层 {layer}"
            )

    plt.axhline(
        y=CONFIG["tau"],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"目标阈值τ={CONFIG['tau']:.2f}"
    )

    plt.xlabel("孕周（周）",fontsize=15)
    plt.ylabel("达到Y≥4%的比例",fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()

    print("\n===== 分析结果汇总 =====")
    print(结果表.to_string(index=False))
    print(f"\n所有结果已保存至：{os.path.abspath(CONFIG['outdir'])}")

if __name__ == "__main__":
    try:
        import lifelines
        import sklearn
    except ImportError as e:
        missing_lib = str(e).split("'")[1]
        print(f"缺少必要的库：{missing_lib}")
        print(f"请先安装：pip install {missing_lib}")
    else:
        main()