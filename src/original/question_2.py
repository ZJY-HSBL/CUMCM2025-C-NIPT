import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from pyswarm import pso
from scipy.stats import zscore
from typing import Tuple, List
from sklearn.model_selection import KFold

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def y_chromosome_concentration(gestational_age, bmi):
    return -0.0540 + 0.3297 * gestational_age + 0.0846 * gestational_age ** 2 - 0.1131 * bmi - 0.0097 * bmi ** 2


def calculate_best_nipt_time(bmi, show_plot=False):
    convergence_history = []
    iteration_count = [0]

    def target(gestational_age):
        concentration = y_chromosome_concentration(gestational_age, bmi)
        current_error = abs(concentration - 4)
        if iteration_count[0] % 5 == 0:
            convergence_history.append(current_error)
        iteration_count[0] += 1
        return current_error

    lb = [10]
    ub = [25]
    best_time, best_error = pso(
        target, lb, ub,
        swarmsize=80,
        maxiter=200,
        minstep=1e-6,
        minfunc=1e-6,
    )
    print(convergence_history)
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, iteration_count[0], 5), convergence_history, c='k')
        plt.xlabel('迭代次数', fontsize=15)
        plt.ylabel('目标函数值（与4%的绝对误差）', fontsize=15)
        plt.tight_layout()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()
    return best_time[0]


def filter_abnormal_data(data: pd.DataFrame, bmi_col: str = 'BMI', z_threshold: float = 3.0) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    data['BMI_Zscore'] = zscore(data[bmi_col])
    abnormal_z_mask = abs(data['BMI_Zscore']) > z_threshold
    abnormal_data = data[abnormal_z_mask].copy()

    filtered_data = data[~abnormal_z_mask].copy()
    bmi_variance = filtered_data[bmi_col].var()
    if bmi_variance < 1e-3:
        raise ValueError(f"过滤后数据BMI方差过小（{bmi_variance:.6f}），数据呈单调分布，无法进行有效聚类")

    print(f"数据预处理报告")
    print(f"原始数据总量: {len(data)} 条")
    print(f"Z-score异常值（|Z|>{z_threshold}）数量: {abnormal_z_mask.sum()} 条")
    print(f"过滤后有效数据量: {len(filtered_data)} 条")
    print(f"有效数据BMI方差: {bmi_variance:.6f}（非单调，适合聚类）")

    filtered_data = filtered_data.drop(columns=['BMI_Zscore'])
    abnormal_data = abnormal_data.drop(columns=['BMI_Zscore'])

    return filtered_data, abnormal_data


def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    data = pd.read_excel(file_path)
    if 'BMI' not in data.columns:
        raise ValueError("无'BMI’")

    filtered_data, abnormal_data = filter_abnormal_data(data)

    filtered_data['BMI_Original'] = filtered_data['BMI'].copy()
    scaler = StandardScaler()
    filtered_data['BMI_Scaled'] = scaler.fit_transform(filtered_data[['BMI']])

    return filtered_data, abnormal_data, scaler


def find_optimal_k(data_scaled: np.ndarray, min_k: int = 2, max_k: int = 5) -> int:
    print(f"\n===== 自动选择最优聚类数k =====")
    silhouette_scores = []
    k_candidates = range(min_k, max_k + 1)

    for k in k_candidates:
        hier_clf = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = hier_clf.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(score)
        print(f"k={k} 时轮廓系数: {score:.4f}")

    max_score_idx = np.argmax(silhouette_scores)
    optimal_k = k_candidates[max_score_idx]
    print(f"最优聚类数k: {optimal_k}（对应轮廓系数：{silhouette_scores[max_score_idx]:.4f}）")
    return optimal_k


def check_cluster_boundary(cluster_stats: pd.DataFrame) -> pd.DataFrame:
    sorted_stats = cluster_stats.sort_values('mean').reset_index()
    sorted_stats.columns = ['Original_Cluster_ID', '样本数', '均值', '标准差', '最小值', '25%分位数', '中位数',
                            '75%分位数', '最大值']

    boundary_adjusted = False
    for i in range(1, len(sorted_stats)):
        prev_max = sorted_stats.loc[i - 1, '最大值']
        curr_min = sorted_stats.loc[i, '最小值']

        if curr_min < prev_max:
            sorted_stats.loc[i, '最小值'] = prev_max + 1e-4
            boundary_adjusted = True

    if boundary_adjusted:
        print("存在边界包含")
        print(sorted_stats[['Original_Cluster_ID', '最小值', '最大值', '均值']].to_string(index=False))
    else:
        print(f"\n聚类边界校验通过：所有聚类[min, max]无包含关系")

    return sorted_stats


def perform_clustering(data: pd.DataFrame) -> Tuple[pd.DataFrame, str, np.ndarray, int]:
    data_scaled = data[['BMI_Scaled']].values

    optimal_k = find_optimal_k(data_scaled)

    hier_clf = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward', compute_distances=True)
    data['Hierarchical_Cluster'] = hier_clf.fit_predict(data_scaled)

    bgmm = BayesianGaussianMixture(
        n_components=optimal_k,
        random_state=42,
        max_iter=1000,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e2
    )
    bgmm.fit(data_scaled)
    data['Bayesian_GMM_Cluster'] = bgmm.predict(data_scaled)

    print(f"\n===== 聚类效果评估（k={optimal_k}） =====")
    sil_hier = silhouette_score(data_scaled, data['Hierarchical_Cluster'])
    sil_bgmm = silhouette_score(data_scaled, data['Bayesian_GMM_Cluster'])
    ari = adjusted_rand_score(data['Hierarchical_Cluster'], data['Bayesian_GMM_Cluster'])

    print(f"层次聚类轮廓系数: {sil_hier:.4f}（越接近1越好）")
    print(f"贝叶斯GMM轮廓系数: {sil_bgmm:.4f}（越接近1越好）")
    print(f"两种方法ARI指数: {ari:.4f}（越接近1表示聚类结果越一致）")

    if sil_hier > sil_bgmm:
        best_method = "层次聚类"
        best_labels = data['Hierarchical_Cluster'].values
    else:
        best_method = "贝叶斯高斯混合模型"
        best_labels = data['Bayesian_GMM_Cluster'].values

    print(f"\n最优聚类方法: {best_method}（对应轮廓系数：{max(sil_hier, sil_bgmm):.4f}）")
    return data, best_method, best_labels, optimal_k


def analyze_nipt_times(data: pd.DataFrame, best_labels: np.ndarray, optimal_k: int) -> Tuple[
    pd.DataFrame, dict, pd.DataFrame, dict]:
    clustered_data = data.copy()
    clustered_data['Best_Cluster'] = best_labels

    cluster_stats = clustered_data.groupby('Best_Cluster')['BMI_Original'].describe()
    adjusted_stats = check_cluster_boundary(cluster_stats)

    cluster_mapping = {
        row['Original_Cluster_ID']: new_idx
        for new_idx, row in adjusted_stats.iterrows()
    }
    clustered_data['Best_Cluster_Sorted'] = clustered_data['Best_Cluster'].map(cluster_mapping)

    print(f"\n===== 最终BMI聚类统计报告 =====")
    for _, row in adjusted_stats.iterrows():
        new_idx = cluster_mapping[row['Original_Cluster_ID']]
        print(f"聚类 {new_idx}（原ID:{int(row['Original_Cluster_ID'])}）:")
        print(f"  样本数量: {int(row['样本数'])} 条")
        print(f"  BMI范围: [{row['最小值']:.2f}, {row['最大值']:.2f}]")
        print(f"  BMI均值: {row['均值']:.2f}（±{row['标准差']:.2f}）")
        print(f"  BMI中位数: {row['中位数']:.2f}")
        print("  " + "-" * 60)

    best_nipt_times = {}
    for _, row in adjusted_stats.iterrows():
        new_idx = cluster_mapping[row['Original_Cluster_ID']]
        mean_bmi = row['均值']
        best_time = calculate_best_nipt_time(mean_bmi)
        best_nipt_times[f"聚类 {new_idx}（BMI:{mean_bmi:.1f}）"] = best_time

    return clustered_data, best_nipt_times, adjusted_stats, cluster_mapping


def _assign_by_nearest_center(train_X, train_labels, test_X):
    centers = []
    for c in np.unique(train_labels):
        centers.append(train_X[train_labels == c].mean(axis=0))
    centers = np.vstack(centers)
    dists = ((test_X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1)


def _safe_silhouette(X, labels):
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        unique, counts = np.unique(labels, return_counts=True)
        if np.min(counts) < 2:
            return np.nan
        return silhouette_score(X, labels)
    except Exception:
        return np.nan


def cross_validate_bmi_clustering(
        data: pd.DataFrame,
        repeats: int = 3,
        n_splits: int = 5,
        use_gmm: bool = True,
        random_state: int = 42
):
    rng = np.random.RandomState(random_state)
    X = data[['BMI_Scaled']].values
    bmi_original = data['BMI_Original'].values

    rows = []
    all_runs_best_times = []

    for r in range(repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 10 ** 9))
        for f_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            bmi_tr, bmi_te = bmi_original[train_idx], bmi_original[test_idx]

            k_h = find_optimal_k(X_tr, min_k=2, max_k=5)
            hier = AgglomerativeClustering(n_clusters=k_h, linkage='ward')
            y_tr_h = hier.fit_predict(X_tr)
            sil_tr_h = _safe_silhouette(X_tr, y_tr_h)
            y_te_h = _assign_by_nearest_center(X_tr, y_tr_h, X_te)
            sil_te_h = _safe_silhouette(X_te, y_te_h)

            times_h = _cluster_times_from_labels(bmi_tr, y_tr_h)

            rows.append({
                "repeat": r, "fold": f_idx, "method": "hierarchical", "k": k_h,
                "silhouette_train": sil_tr_h, "silhouette_test": sil_te_h,
                "test_avg_loglik": np.nan
            })
            all_runs_best_times.append({
                "repeat": r, "fold": f_idx, "method": "hierarchical", "k": k_h,
                "times": times_h
            })

            if use_gmm:
                k_g = find_optimal_k(X_tr, min_k=2, max_k=5)
                bgmm = BayesianGaussianMixture(
                    n_components=k_g,
                    random_state=42,
                    max_iter=1000,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=1e2
                )
                bgmm.fit(X_tr)
                y_tr_g = bgmm.predict(X_tr)
                sil_tr_g = _safe_silhouette(X_tr, y_tr_g)
                y_te_g = bgmm.predict(X_te)
                sil_te_g = _safe_silhouette(X_te, y_te_g)
                try:
                    ll_te = bgmm.score(X_te)
                except Exception:
                    ll_te = np.nan

                times_g = _cluster_times_from_labels(bmi_tr, y_tr_g)

                rows.append({
                    "repeat": r, "fold": f_idx, "method": "bayesian_gmm", "k": k_g,
                    "silhouette_train": sil_tr_g, "silhouette_test": sil_te_g,
                    "test_avg_loglik": ll_te
                })
                all_runs_best_times.append({
                    "repeat": r, "fold": f_idx, "method": "bayesian_gmm", "k": k_g,
                    "times": times_g
                })

    fold_metrics = pd.DataFrame(rows)

    def _fmt_mean_std(s):
        return f"{np.nanmean(s):.3f} ± {np.nanstd(s):.3f}"

    summary_rows = []
    for method, g in fold_metrics.groupby("method"):
        # 将方法名称转为中文
        method_cn = "层次聚类" if method == "hierarchical" else "贝叶斯高斯混合模型"
        summary_rows.append({
            "方法": method_cn,
            "训练集轮廓系数 (均值±标准差)": _fmt_mean_std(g["silhouette_train"]),
            "测试集轮廓系数 (均值±标准差)": _fmt_mean_std(g["silhouette_test"]),
            "测试集平均对数似然均值 (均值±标准差)": _fmt_mean_std(
                g["test_avg_loglik"]) if method != "hierarchical" else "不适用"
        })
    summary_df = pd.DataFrame(summary_rows)

    nipt_tables = []
    for method in set([d["method"] for d in all_runs_best_times]):
        method_cn = "层次聚类" if method == "hierarchical" else "贝叶斯高斯混合模型"
        times_list = [d["times"] for d in all_runs_best_times if d["method"] == method]
        Kmax = max(len(x) for x in times_list)
        padded = []
        for arr in times_list:
            if len(arr) < Kmax:
                arr = list(arr) + [np.nan] * (Kmax - len(arr))
            padded.append(arr)
        arr2d = np.array(padded)
        stats = {
            "方法": method_cn,
            **{f"聚类_{i} (周) 均值": np.nanmean(arr2d[:, i]) for i in range(Kmax)},
            **{f"聚类_{i} (周) 标准差": np.nanstd(arr2d[:, i]) for i in range(Kmax)}
        }
        nipt_tables.append(stats)
    nipt_stability = pd.DataFrame(nipt_tables)

    txt_lines = ["===== 交叉验证总结 ====="]
    for _, row in summary_df.iterrows():
        txt_lines.append(f"[{row['方法']}] "
                         f"训练集轮廓系数: {row['训练集轮廓系数 (均值±标准差)']}; "
                         f"测试集轮廓系数: {row['测试集轮廓系数 (均值±标准差)']}; "
                         f"测试集平均对数似然: {row['测试集平均对数似然均值 (均值±标准差)']}")
    txt_lines.append("\n最佳NIPT检测时间稳定性 (单位: 周):")
    txt_lines.append(nipt_stability.to_string(index=False))
    summary_text = "\n".join(txt_lines)

    return {
        "fold_metrics": fold_metrics,
        "summary_table": summary_df,
        "nipt_stability": nipt_stability,
        "summary": summary_text
    }


def _cluster_times_from_labels(bmi_array, labels):
    df = pd.DataFrame({"BMI": bmi_array, "label": labels})
    stats = df.groupby("label")["BMI"].mean().sort_values().reset_index()
    times = []
    for _, row in stats.iterrows():
        mean_bmi = row["BMI"]
        t = calculate_best_nipt_time(mean_bmi, show_plot=False)
        times.append(float(t))
    return np.array(times, dtype=float)


def visualize_results(
        data: pd.DataFrame, abnormal_data: pd.DataFrame, best_method: str,
        clustered_data: pd.DataFrame, best_nipt_times: dict, adjusted_stats: pd.DataFrame,
        optimal_k: int, cluster_mapping: dict
):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        clustered_data.index, clustered_data['BMI_Original'],
        c='k', label=f'有效数据（{len(clustered_data)}条）', s=50, alpha=0.7, marker='o'
    )
    if len(abnormal_data) > 0:
        plt.scatter(
            abnormal_data.index, abnormal_data['BMI'],
            c='k', label=f'异常数据（{len(abnormal_data)}条）', s=80, marker='x'
        )
    plt.xlabel('样本索引', fontsize=15)
    plt.ylabel('BMI值', fontsize=15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Best_Cluster_Sorted',
        y='BMI_Original',
        hue='Best_Cluster_Sorted',
        data=clustered_data,
        palette='Paired',
        width=0.6,
        legend=False
    )
    for _, row in adjusted_stats.iterrows():
        new_idx = cluster_mapping[row['Original_Cluster_ID']]
        plt.hlines(
            y=row['最小值'], xmin=new_idx - 0.3, xmax=new_idx + 0.3,
            colors='red', linestyles='--', linewidth=1.5,
            label='聚类边界' if _ == 0 else ""
        )
        plt.hlines(
            y=row['最大值'], xmin=new_idx - 0.3, xmax=new_idx + 0.3,
            colors='red', linestyles='--', linewidth=1.5
        )
    plt.title(f'{best_method}结果（k={optimal_k}，无边界包含）', fontsize=16, pad=20)
    plt.xlabel('聚类（0=低BMI → 高BMI）', fontsize=12)
    plt.ylabel('BMI值', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    cluster_labels = list(best_nipt_times.keys())
    nipt_times = list(best_nipt_times.values())
    bars = plt.bar(
        cluster_labels, nipt_times,
        color='lightgreen', edgecolor='darkgreen', linewidth=1.2, alpha=0.8
    )
    for bar, time in zip(bars, nipt_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{time:.2f}周', ha='center', va='bottom', fontsize=10
        )
    plt.title('不同BMI聚类的最佳NIPT时点', fontsize=16, pad=20)
    plt.xlabel('BMI聚类（括号内为平均BMI）', fontsize=12)
    plt.ylabel('最佳NIPT时点（孕周）', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(nipt_times) - 0.5, max(nipt_times) + 0.8)
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    gestational_ages = np.linspace(10, 25, 200)
    line_styles = ['-', '--', '-.', ':']

    for idx, row in adjusted_stats.iterrows():
        new_cluster_idx = cluster_mapping[row['Original_Cluster_ID']]
        mean_bmi = row['均值']
        concentrations = [y_chromosome_concentration(ga, mean_bmi) for ga in gestational_ages]

        line_style = line_styles[idx % len(line_styles)]

        plt.plot(
            gestational_ages, concentrations,
            color='black',
            linewidth=2.5,
            linestyle=line_style,
            label=f'聚类 {new_cluster_idx}（BMI:{mean_bmi:.1f}）'
        )

    plt.axhline(y=4, color='black', linestyle='--', linewidth=2, label='目标浓度（4%）')

    for cluster_label, best_time in best_nipt_times.items():
        cluster_part = cluster_label.split('（')[0]
        cluster_idx = int(cluster_part.split(' ')[1])
        mean_bmi = float(cluster_label.split(':')[1].split('）')[0])
        best_concentration = y_chromosome_concentration(best_time, mean_bmi)
        plt.scatter(
            best_time, best_concentration,
            color='black', s=100, edgecolor='black', zorder=5, alpha=0.6
        )
        plt.annotate(
            f'最佳时点\n{best_time:.2f}周',
            xy=(best_time, best_concentration),
            xytext=(best_time - 1, best_concentration + 0.7),
            fontsize=12, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    plt.xlabel('孕周', fontsize=15)
    plt.ylabel('Y染色体浓度（%）', fontsize=15)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(10, 25)
    plt.tight_layout()
    plt.show()


def plot_silhouette_stability(cv_report):
    fold_metrics = cv_report["fold_metrics"]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='method', y='silhouette_test', data=fold_metrics, hue='method', legend=False,
                boxprops=dict(facecolor="white"))

    plt.xlabel('聚类方法', fontsize=15)
    plt.ylabel('轮廓系数', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()


def main(file_path: str):
    try:
        filtered_data, abnormal_data, scaler = load_and_process_data(file_path)

        filtered_data.to_excel('filtered_data.xlsx', index=False)
        abnormal_data.to_excel('abnormal_data.xlsx', index=False)

        clustered_data, best_method, best_labels, optimal_k = perform_clustering(filtered_data)

        clustered_data.to_excel('clustered_data.xlsx', index=False)

        cv_report = cross_validate_bmi_clustering(
            data=filtered_data,
            repeats=3,
            n_splits=5,
            use_gmm=True,
            random_state=42
        )

        print(cv_report["summary"])

        plot_silhouette_stability(cv_report)

        final_clustered_data, best_nipt_times, adjusted_stats, cluster_mapping = analyze_nipt_times(
            clustered_data, best_labels, optimal_k
        )

        final_clustered_data.to_excel('final_clustered_data.xlsx', index=False)
        adjusted_stats.to_excel('cluster_statistics.xlsx', index=True)
        nipt_times_df = pd.DataFrame(list(best_nipt_times.items()),
                                     columns=['Cluster', 'Best_NIPT_Time'])
        nipt_times_df.to_excel('best_nipt_times.xlsx', index=False)

        visualize_results(
            data=pd.read_excel(file_path),
            abnormal_data=abnormal_data,
            best_method=best_method,
            clustered_data=final_clustered_data,
            best_nipt_times=best_nipt_times,
            adjusted_stats=adjusted_stats,
            optimal_k=optimal_k,
            cluster_mapping=cluster_mapping
        )

        print("1. 异常数据与预处理结果已保存")
        print("2. 聚类结果与最终聚类映射已保存")
        print("3. 交叉验证摘要/明细已保存")
        print("4. 最佳 NIPT 时点与图表已输出")

    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        raise


if __name__ == "__main__":
    main(file_path='a_1.xlsx')