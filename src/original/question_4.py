import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import re
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# 全局配置
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12


class FetalAbnormalityAnalyzer:
    def __init__(self):
        self.header_mapping = {
            '序号': 'serial_no',
            '孕妇代码': 'pregnant_id',
            '年龄': 'age',
            '身高': 'height',
            '体重': 'weight',
            '孕妇BMI': 'bmi',
            '检测孕周': 'gestational_week',
            '在参考基因组上比对的比例': 'alignment_ratio',
            '被过滤掉读段数的比例': 'filtered_ratio',
            '13号染色体的Z值': 'z_13',
            '18号染色体的Z值': 'z_18',
            '21号染色体的Z值': 'z_21',
            'X染色体的Z值': 'z_x',
            'X染色体浓度': 'x_chromosome_concentration',
            '13号染色体的GC含量': 'gc_13',
            '18号染色体的GC含量': 'gc_18',
            '21号染色体的GC含量': 'gc_21',
            '染色体的非整倍体': 'aneuploidy_label',  # 0=正常,1=异常
            '胎儿是否健康': 'health_label'  # 0=不健康,1=健康
        }

        self.required_fields = [
            'pregnant_id', 'bmi', 'alignment_ratio', 'filtered_ratio',
            'z_13', 'z_18', 'z_21', 'z_x',
            'gc_13', 'gc_18', 'gc_21'
        ]
        # 风险阈值
        self.high_risk_score = 0.60
        self.mid_risk_low = 0.40

        # 初始化工具   数据存储
        self.scaler = StandardScaler()
        self.model = None
        self.model_features = None
        self.raw_data = None
        self.processed_data = None
        self.final_result = None
        self.X_scaled = None  # 标准化特征矩阵
        self.current_hi_thr = None  # 当前使用的高风险阈值
        self.current_mid_lo = None  # 当前使用的中风险阈值

    # Excel
    def load_excel(self, file_path: str) -> pd.DataFrame:
        print(f"[1/7] 加载Excel数据：{os.path.basename(file_path)}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        if not file_path.endswith(('.xlsx', '.xls')):
            raise ValueError("仅支持.xlsx/.xls格式")

        # 读取数据
        try:
            raw_df = pd.read_excel(file_path, header=0)
            print(f"成功读取：{len(raw_df)}个样本，{len(raw_df.columns)}个字段")
        except Exception as e:
            raise RuntimeError(f"读取失败：{str(e)}")

        # 表头映射
        mapped_df = raw_df.rename(columns=self.header_mapping)

        # 打印缺失率
        print("\n数据诊断报告：")
        for field in self.required_fields + ['aneuploidy_label', 'health_label']:
            if field in mapped_df.columns:
                miss_rate = mapped_df[field].isna().mean() * 100
                print(f"  {field:20} 缺失率：{miss_rate:.1f}%")

        # 标签预处理（统一格式）
        def norm_str(s):
            if pd.isna(s):
                return s
            return str(s).strip().lower()

        # 保存原始标签用于对比
        orig_health = mapped_df['health_label'].copy() if 'health_label' in mapped_df.columns else None
        orig_aneu = mapped_df['aneuploidy_label'].copy() if 'aneuploidy_label' in mapped_df.columns else None

        # 处理染色体非整倍体标签
        def map_aneuploidy(v):
            if pd.isna(v):
                return pd.NA
            s = norm_str(v)
            # 包含T13/T18/T21视为异常
            if any(x in s for x in ['t13', 't18', 't21']):
                return 1
            # 正常标签映射为0
            normal_tags = {'正常', '无', '否', '阴性', 'low', '0', '未检出'}
            if any(tag in s for tag in normal_tags):
                return 0
            # 异常标签映射为1
            abnormal_tags = {'异常', '有', '是', '阳性', 'high', '1', '检出', '可疑'}
            if any(tag in s for tag in abnormal_tags):
                return 1
            # 数字格式处理
            try:
                return 1 if float(s) == 1.0 else 0
            except:
                return pd.NA

        # 处理胎儿健康标签
        def map_health(v):
            if pd.isna(v):
                return pd.NA
            s = norm_str(v)
            healthy_tags = {'健康', '正常', '1', 'yes', '阴性'}
            if any(tag in s for tag in healthy_tags):
                return 1
            unhealthy_tags = {'不健康', '异常', '0', 'no', '阳性'}
            if any(tag in s for tag in unhealthy_tags):
                return 0
            try:
                return 1 if float(s) == 1.0 else 0
            except:
                return pd.NA

        # 应用标签映射
        if 'aneuploidy_label' in mapped_df.columns:
            mapped_df['aneuploidy_label'] = mapped_df['aneuploidy_label'].apply(map_aneuploidy).astype('Int64')
        if 'health_label' in mapped_df.columns:
            mapped_df['health_label'] = mapped_df['health_label'].apply(map_health).astype('Int64')

        # 打印标签分布
        if orig_health is not None:
            print("\nhealth_label原始分布:")
            print(orig_health.value_counts(dropna=False))
            print("health_label映射后分布:")
            print(mapped_df['health_label'].value_counts(dropna=False))
        if orig_aneu is not None:
            print("\naneuploidy_label映射后分布:")
            print(mapped_df['aneuploidy_label'].value_counts(dropna=False))

        self.raw_data = raw_df
        mapped_df = mapped_df.reset_index(drop=True)
        return mapped_df

    # 2. 数据预处理
    def preprocess_data(self, mapped_df: pd.DataFrame) -> pd.DataFrame:
        print("\n[2/7] 数据预处理")
        df = mapped_df.copy()
        initial_total = len(df)

        # 1. 过滤低质量样本（比对比例≥0.7，过滤比例≤0.05）
        quality_mask = (df['alignment_ratio'] >= 0.7) & (df['filtered_ratio'] <= 0.05)
        dropped = len(df) - quality_mask.sum()
        df = df[quality_mask].copy()
        print(f"1. 质量过滤：剔除{dropped}个样本（{initial_total}→{len(df)}）")

        # 2. 处理缺失值
        print("\n2. 缺失值填充：")
        for field in self.required_fields:
            if field not in df.columns:
                continue
            miss_count = df[field].isna().sum()
            if miss_count == 0:
                print(f"  {field:20} 无缺失")
                continue

            # 不同字段用不同填充方式
            if field == 'bmi':
                # BMI用同孕周均值填充（没有则用全局均值）
                if 'gestational_week' in df.columns:
                    df['bmi'] = df.groupby('gestational_week')['bmi'].transform(lambda x: x.fillna(x.mean()))
                df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
                print(f"  {field:20} 填充{miss_count}个（同孕周均值→全局均值）")
            elif field in ['z_13', 'z_18', 'z_21', 'z_x', 'gc_13', 'gc_18', 'gc_21']:
                df[field] = df[field].fillna(df[field].mean())
                print(f"  {field:20} 填充{miss_count}个（全局均值）")
            elif field in ['alignment_ratio', 'filtered_ratio']:
                df[field] = df[field].fillna(df[field].median())
                print(f"  {field:20} 填充{miss_count}个（中位数）")

        # 3. 特征工程（衍生关键特征）
        print("\n3. 特征工程：")
        # 多染色体Z值异常数（Z绝对值>3算异常）
        df['multi_z_abnormal'] = (
            (df['z_13'].abs() > 3).astype(int) +
            (df['z_18'].abs() > 3).astype(int) +
            (df['z_21'].abs() > 3).astype(int) +
            (df['z_x'].abs() > 3).astype(int)
        )
        # GC含量异常标记（<40%或>60%算异常）
        df['gc_abnormal_flag'] = (
            (df['gc_13'] < 40) | (df['gc_13'] > 60) |
            (df['gc_18'] < 40) | (df['gc_18'] > 60) |
            (df['gc_21'] < 40) | (df['gc_21'] > 60)
        ).astype(int)
        # BMI分组（按标准划分）
        df['bmi_group'] = pd.cut(
            df['bmi'], bins=[-np.inf, 18.5, 24, 30, np.inf],
            labels=[0, 1, 2, 3], right=True
        ).astype(int)
        print("  衍生特征：multi_z_abnormal, gc_abnormal_flag, bmi_group")

        # 4. 确定模型输入特征
        self.model_features = [
            'z_13', 'z_18', 'z_21', 'z_x',
            'gc_13', 'gc_18', 'gc_21',
            'alignment_ratio', 'filtered_ratio',
            'multi_z_abnormal', 'gc_abnormal_flag',
            'bmi_group'
        ]
        # 过滤掉不存在的特征
        self.model_features = [f for f in self.model_features if f in df.columns]
        print(f"\n4. 模型输入特征：{', '.join(self.model_features)}（共{len(self.model_features)}个）")

        self.processed_data = df
        print(f"预处理完成：保留{len(df)}个有效样本")
        return df

    # 3. 特征有效性分析（绘图）
    def analyze_feature_effectiveness(self, df: pd.DataFrame) -> None:
        print("\n[3/7] 特征有效性分析")
        # 创建2x2子图
        plt.figure(figsize=(15, 12))

        # 子图1：4条染色体Z值分布
        plt.subplot(2, 2, 1)
        for feat in ['z_13', 'z_18', 'z_21', 'z_x']:
            if feat in df.columns:
                sns.histplot(df[feat], kde=True, label=feat, alpha=0.6)
        plt.title('4条染色体Z值分布', fontweight='bold')
        plt.xlabel('Z值')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图2：3条染色体GC含量分布
        plt.subplot(2, 2, 2)
        for feat in ['gc_13', 'gc_18', 'gc_21']:
            if feat in df.columns:
                sns.histplot(df[feat], kde=True, label=feat, alpha=0.6)
        plt.title('3条染色体GC含量分布', fontweight='bold')
        plt.xlabel('GC含量（%）')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图3：孕妇BMI分布
        plt.subplot(2, 2, 3)
        if 'bmi' in df.columns:
            sns.histplot(df['bmi'], kde=True, alpha=0.7, color='lightblue')
            # 标注BMI标准线
            plt.axvline(x=18.5, linestyle='--', color='red', label='偏瘦(18.5)')
            plt.axvline(x=24, linestyle='--', color='green', label='正常(24)')
            plt.axvline(x=30, linestyle='--', color='orange', label='肥胖(30)')
        plt.title('孕妇BMI分布', fontweight='bold')
        plt.xlabel(r'BMI（kg/m$^2$）')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图4：参考基因组比对比例分布
        plt.subplot(2, 2, 4)
        if 'alignment_ratio' in df.columns:
            sns.histplot(df['alignment_ratio'], kde=True, alpha=0.7, color='lightgreen')
            plt.axvline(x=0.7, linestyle='--', color='red', label='合格线(0.7)')
        plt.title('参考基因组比对比例分布', fontweight='bold')
        plt.xlabel('比对比例')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(alpha=0.3)

        # 保存图片
        plot_path = '特征分布分析图.png'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  特征分布图已保存：{os.path.abspath(plot_path)}")

    # 4. 训练孤立森林模型
    def train_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[4/7] 训练孤立森林模型")
        # 准备特征数据
        X = df[self.model_features].copy()
        # 填充剩余缺失值
        for feat in self.model_features:
            if X[feat].isna().any():
                X[feat] = X[feat].fillna(X[feat].mean())

        # 检查样本数
        if len(X) < 10:
            raise RuntimeError(f"样本数不足（仅{len(X)}个），至少需要10个样本")
        print(f"模型输入：{len(X)}个样本 × {len(self.model_features)}个特征")

        # 特征标准化
        self.X_scaled = self.scaler.fit_transform(X)
        print("特征标准化完成（均值≈0，标准差≈1）")

        # 训练模型
        self.model = IsolationForest(
            n_estimators=150,  # 树的数量
            max_samples=min(0.25, len(X)),  # 每棵树用的样本比例
            contamination=0.1,  # 异常样本比例假设
            random_state=42,
            n_jobs=-1  # 用所有CPU核心
        )
        self.model.fit(self.X_scaled)

        # 计算异常得分（0-1之间，越大越异常）
        decision_scores = self.model.decision_function(self.X_scaled)
        score_range = decision_scores.max() - decision_scores.min()
        if score_range == 0:
            anomaly_scores = np.zeros_like(decision_scores)
        else:
            anomaly_scores = (decision_scores.max() - decision_scores) / score_range

        # 保存得分到数据框
        df2 = df.copy()
        df2['异常得分'] = np.round(anomaly_scores, 4)
        print(f"模型训练完成：异常得分范围 {anomaly_scores.min():.4f} ~ {anomaly_scores.max():.4f}")
        return df2

    # 5. 风险等级判定
    def classify_risk_level(self, df_with_score: pd.DataFrame) -> pd.DataFrame:
        print("\n[5/7] 三级风险判定")
        df = df_with_score.copy()
        scores = df['异常得分'].astype(float)

        # 1. 确定风险阈值（优先用分位数，否则用默认）
        hi_thr = None
        mid_lo = None
        if scores.notna().sum() >= 50:
            # 计算分位数（Q90=高风险，Q60=中风险下限）
            q90 = float(np.quantile(scores.dropna(), 0.90))
            q60 = float(np.quantile(scores.dropna(), 0.60))
            if q90 > q60:
                hi_thr = q90
                mid_lo = q60
                print(f"  使用分位数阈值：高风险≥{hi_thr:.2f}（Q90），中风险{mid_lo:.2f}~{hi_thr:.2f}（Q60~Q90）")
            else:
                print("  分位数阈值不稳定，使用默认阈值")
        # 用默认阈值
        if hi_thr is None or mid_lo is None:
            hi_thr = self.high_risk_score
            mid_lo = self.mid_risk_low
            print(f"  使用默认阈值：高风险≥{hi_thr:.2f}，中风险{mid_lo:.2f}~{hi_thr:.2f}")

        # 保存当前阈值
        self.current_hi_thr = hi_thr
        self.current_mid_lo = mid_lo

        # 2. 标记风险等级
        # 高风险条件
        df['高风险标记'] = (df['异常得分'] >= hi_thr).astype(int)
        # 中风险条件（得分+临床规则）
        df['中风险标记'] = (
            ((df['异常得分'] >= mid_lo) & (df['异常得分'] < hi_thr)) |
            (df['multi_z_abnormal'] == 2) |  # 2个染色体Z值异常
            (df['z_x'] < -2.5) |  # X染色体Z值过低
            ((df['bmi'] >= 30) & (df['bmi'] <= 35))  # BMI超重
        ).astype(int)
        # 低风险（默认）
        df['低风险标记'] = 0

        # 3. 分配最终风险等级
        def assign_risk(row):
            if row['高风险标记'] == 1:
                return '高风险'
            elif row['中风险标记'] == 1:
                return '中风险'
            else:
                return '低风险'

        df['风险等级'] = df.apply(assign_risk, axis=1)

        # 4. 打印风险分布
        risk_count = df['风险等级'].value_counts()
        risk_pct = (df['风险等级'].value_counts(normalize=True) * 100).round(2)
        print("\n风险等级分布：")
        for level in ['高风险', '中风险', '低风险']:
            print(f"  {level}：{risk_count.get(level, 0)}个样本（{risk_pct.get(level, 0)}%）")

        self.final_result = df
        return df

    # 6. 模型验证
    def validate_model(self, df: pd.DataFrame):
        print("\n[6/7] 模型验证")
        label_col = 'aneuploidy_label'
        # 检查是否有有效标签
        if label_col not in df.columns or not df[label_col].notna().any():
            print("  无有效标签（aneuploidy_label全缺失），跳过验证")
            return

        # 筛选有标签的样本
        labeled_df = df[df[label_col].notna()].copy()
        true_labels = labeled_df[label_col].values

        # 情况1：只有一类标签（全正常或全异常）
        if len(np.unique(true_labels)) < 2:
            # 用高+中风险作为异常预测
            pred_labels = labeled_df['风险等级'].isin(['高风险', '中风险']).astype(int).values
            recall = (pred_labels == 1).mean() * 100.0
            print(f"仅单类标签可用：阳性召回率≈{recall:.2f}%（{pred_labels.sum()}/{len(pred_labels)}）")

            # 阈值扫描（找最优阈值）
            pos_indices = labeled_df.index[true_labels == 1]
            thresholds = np.linspace(0.20, 0.80, 13)
            print("\n阈值扫描结果（阈值→阳性召回率→命中率）：")
            best_thr = None
            best_recall = 0
            for thr in thresholds:
                pred = (df['异常得分'] >= thr).astype(int)
                # 计算阳性召回率
                if len(pos_indices) > 0:
                    pos_recall = (pred.loc[pos_indices] == 1).mean() * 100.0
                else:
                    pos_recall = 0.0
                # 计算命中率
                if pred.sum() > 0:
                    hit_rate = (df.loc[pred == 1, label_col].eq(1)).mean() * 100.0
                else:
                    hit_rate = 0.0
                # 记录最优阈值
                if pos_recall > best_recall:
                    best_recall = pos_recall
                    best_thr = thr
                print(f"  {thr:.2f} → {pos_recall:.2f}% → {hit_rate:.2f}%")
            if best_thr is not None:
                print(f"\n建议最优阈值：{best_thr:.2f}（召回率≈{best_recall:.1f}%）")
            return

        # 情况2：有两类标签（正常+异常）
        labeled_df['二元预测'] = labeled_df['风险等级'].isin(['高风险', '中风险']).astype(int)
        # 混淆矩阵
        cm = confusion_matrix(labeled_df[label_col], labeled_df['二元预测'])
        print("\n混淆矩阵（行=真实标签，列=预测标签）：")
        print(cm)
        # 分类报告（精确率、召回率、F1）
        print("\n分类性能报告：")
        print(classification_report(
            labeled_df[label_col], labeled_df['二元预测'],
            target_names=['正常', '异常'], zero_division=0
        ))

    # 7. 保存结果到Excel
    def save_final_results(self, output_filename: str = "女胎异常判定结果.xlsx"):
        print("\n[7/7] 保存结果到Excel")
        if self.final_result is None:
            raise RuntimeError("无结果可保存，请先完成风险判定")

        # 合并原始数据和分析结果
        raw_df = self.raw_data.copy().reset_index(drop=True)
        result_df = self.final_result.copy().reset_index(drop=True)

        # 要保存的分析字段
        analysis_fields = ['异常得分', '风险等级', 'multi_z_abnormal', 'gc_abnormal_flag']
        analysis_fields = [f for f in analysis_fields if f in result_df.columns]
        # 合并数据
        final_df = pd.concat([raw_df, result_df[analysis_fields]], axis=1)
        # 重命名字段（更易懂）
        final_df = final_df.rename(columns={
            'multi_z_abnormal': '多染色体Z值异常数',
            'gc_abnormal_flag': 'GC异常标记（0=无，1=有）'
        })

        # 创建Excel并设置样式
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "判定结果"
        # 写入数据
        for row in dataframe_to_rows(final_df, index=False, header=True):
            ws1.append(row)

        # 样式设置
        # 表头样式
        header_font = Font(name='SimHei', size=11, bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
        # 风险等级颜色（高=浅红，中=浅黄，低=浅绿）
        high_fill = PatternFill(start_color='FFE6E6', end_color='FFE6E6', fill_type='solid')
        mid_fill = PatternFill(start_color='FFFFE6', end_color='FFFFE6', fill_type='solid')
        low_fill = PatternFill(start_color='E6FFE6', end_color='E6FFE6', fill_type='solid')
        # 边框
        border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        # 应用表头样式
        for cell in ws1[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = border

        # 找到风险等级列并着色
        risk_col = None
        for col in ws1.iter_cols(1, ws1.max_column):
            if col[0].value == '风险等级':
                risk_col = col[0].column_letter
                break
        if risk_col is not None:
            for row in range(2, ws1.max_row + 1):
                cell = ws1[f"{risk_col}{row}"]
                if cell.value == '高风险':
                    cell.fill = high_fill
                elif cell.value == '中风险':
                    cell.fill = mid_fill
                elif cell.value == '低风险':
                    cell.fill = low_fill
                cell.border = border

        # 列宽自适应
        for column in ws1.columns:
            max_len = 0
            col_letter = column[0].column_letter
            for cell in column:
                try:
                    max_len = max(max_len, len(str(cell.value)))
                except:
                    pass
            ws1.column_dimensions[col_letter].width = min(max_len + 2, 50)  # 限制最大宽度

        # 添加结果说明表
        ws2 = wb.create_sheet(title="结果说明")
        explanation = [
            ['字段名称', '说明', '备注'],
            ['异常得分', '0-1之间，越大表示异常可能性越高', '基于孤立森林模型'],
            ['风险等级', '高/中/低风险', '结合分位数和临床规则'],
            ['多染色体Z值异常数', 'Z值绝对值>3的染色体数量（0-4）', '含13/18/21/X染色体'],
            ['GC异常标记', '0=无异常，1=GC含量<40%或>60%', 'GC异常影响测序质量'],
            ['孕妇BMI', '身体质量指数', '分组：<18.5(偏瘦)、18.5-24(正常)、24-30(超重)、≥30(肥胖)']
        ]
        # 写入说明
        for row in explanation:
            ws2.append(row)
        # 说明表表头样式
        for cell in ws2[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = border
        # 说明表列宽
        for column in ws2.columns:
            max_len = 0
            col_letter = column[0].column_letter
            for cell in column:
                try:
                    max_len = max(max_len, len(str(cell.value)))
                except:
                    pass
            ws2.column_dimensions[col_letter].width = min(max_len + 2, 60)

        # 保存文件
        wb.save(output_filename)
        print(f"结果文件已保存：{os.path.abspath(output_filename)}")

    # 灵敏度分析1：特征灵敏度（特征变化对得分的影响）
    def analyze_feature_sensitivity(self, perturbation=0.01):
        print("\n[灵敏度分析1] 特征灵敏度")
        if self.model is None or self.X_scaled is None:
            raise RuntimeError("请先训练模型")

        n_samples, n_feats = self.X_scaled.shape
        sens_results = []

        # 计算原始异常得分
        raw_decision = self.model.decision_function(self.X_scaled)
        score_range = raw_decision.max() - raw_decision.min()
        if score_range == 0:
            raw_scores = np.zeros_like(raw_decision)
        else:
            raw_scores = (raw_decision.max() - raw_decision) / score_range

        # 灵敏度
        for i, feat in enumerate(self.model_features):
            X_perturb = self.X_scaled.copy()
            feat_std = np.std(X_perturb[:, i])
            X_perturb[:, i] += perturbation * feat_std

            # 扰动后的得分
            perturb_decision = self.model.decision_function(X_perturb)
            if score_range == 0:
                perturb_scores = np.zeros_like(perturb_decision)
            else:
                perturb_scores = (raw_decision.max() - perturb_decision) / score_range

            # 得分平均绝对变化
            score_diff = np.mean(np.abs(perturb_scores - raw_scores))
            sens_results.append((feat, score_diff))
            print(f"  特征 {feat:20} 灵敏度：{score_diff:.6f}")

        sens_df = pd.DataFrame(sens_results, columns=['特征', '灵敏度'])
        sens_df = sens_df.sort_values('灵敏度', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='灵敏度', y='特征', data=sens_df, color='lightcoral')
        plt.title('特征灵敏度分析（值越大影响越显著）', fontweight='bold')
        plt.xlabel('灵敏度')
        plt.ylabel('特征')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('特征灵敏度分析图.png', dpi=300)
        plt.close()
        print(f"  特征灵敏度图已保存：{os.path.abspath('特征灵敏度分析图.png')}")

    # 2：阈值
    def analyze_threshold_sensitivity(self):
        print("\n[灵敏度分析2] 阈值灵敏度")
        if self.final_result is None:
            raise RuntimeError("请先完成风险分类")

        df = self.final_result.copy()
        base_hi = self.current_hi_thr
        base_mid = self.current_mid_lo

        # 1. 高风险阈值变化分析（±20%范围）
        hi_thrs = np.linspace(base_hi * 0.8, base_hi * 1.2, 10)
        high_risk_counts = []
        for thr in hi_thrs:
            # 高风险样本数
            count = ((df['异常得分'] >= thr) |
                     (df['multi_z_abnormal'] == 2) |
                     (df['z_x'] < -2.5) |
                     ((df['bmi'] >= 30) & (df['bmi'] <= 35))).sum()
            high_risk_counts.append(count)

        # 高风险
        plt.figure(figsize=(8, 5))
        plt.plot(hi_thrs, high_risk_counts, 'o-', color='red')
        plt.axvline(x=base_hi, linestyle='--', color='black', label=f'基准阈值={base_hi:.2f}')
        plt.title('高风险阈值变化对样本数的影响', fontweight='bold')
        plt.xlabel('高风险阈值')
        plt.ylabel('高风险样本数')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('高风险阈值灵敏度分析图.png', dpi=300)
        plt.close()
        print(f"  高风险阈值图已保存：{os.path.abspath('高风险阈值灵敏度分析图.png')}")

        # 2. 中风险   ±20%范围
        mid_thrs = np.linspace(base_mid * 0.8, base_mid * 1.2, 10)
        mid_risk_counts = []
        for thr in mid_thrs:
            # 中风险样本数
            count = ((df['异常得分'] >= thr) & (df['异常得分'] < base_hi) |
                     (df['multi_z_abnormal'] == 2) |
                     (df['z_x'] < -2.5) |
                     ((df['bmi'] >= 30) & (df['bmi'] <= 35))).sum()
            mid_risk_counts.append(count)

        # 中风险
        plt.figure(figsize=(8, 5))
        plt.plot(mid_thrs, mid_risk_counts, 'o-', color='orange')
        plt.axvline(x=base_mid, linestyle='--', color='black', label=f'基准阈值={base_mid:.2f}')
        plt.title('中风险阈值变化对样本数的影响', fontweight='bold')
        plt.xlabel('中风险阈值')
        plt.ylabel('中风险样本数')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('中风险阈值灵敏度分析图.png', dpi=300)
        plt.close()
        print(f"  中风险阈值图已保存：{os.path.abspath('中风险阈值灵敏度分析图.png')}")

    # 3：模型参数灵敏度，    contamination参数影响
    def analyze_model_param_sensitivity(self):
        print("\n[灵敏度分析3] 模型参数灵敏度")
        if self.processed_data is None:
            raise RuntimeError("请先完成数据预处理")

        df = self.processed_data.copy()
        # 准备特征数据
        X = df[self.model_features].copy()
        for feat in self.model_features:
            if X[feat].isna().any():
                X[feat] = X[feat].fillna(X[feat].mean())
        X_scaled = self.scaler.transform(X)

        # 测试不同contamination值
        contamination_vals = [0.05, 0.1, 0.15, 0.2, 0.25]
        high_risk_rates = []

        for c in contamination_vals:
            # 临时模型
            temp_model = IsolationForest(
                n_estimators=150,
                max_samples=min(0.25, len(X)),
                contamination=c,
                random_state=42,
                n_jobs=-1
            )
            temp_model.fit(X_scaled)

            # 异常得分
            decision_scores = temp_model.decision_function(X_scaled)
            score_range = decision_scores.max() - decision_scores.min()
            if score_range == 0:
                anomaly_scores = np.zeros_like(decision_scores)
            else:
                anomaly_scores = (decision_scores.max() - decision_scores) / score_range

            # 高风险比例
            high_risk_rate = (anomaly_scores >= self.current_hi_thr).mean() * 100
            high_risk_rates.append(high_risk_rate)
            print(f"  contamination={c:<4} → 高风险比例：{high_risk_rate:.2f}%")

        # 参数影响
        plt.figure(figsize=(8, 5))
        plt.plot(contamination_vals, high_risk_rates, 'o-', color='blue')
        plt.title('IsolationForest contamination参数对高风险比例的影响', fontweight='bold')
        plt.xlabel('contamination（异常样本比例假设）')
        plt.ylabel('高风险样本比例（%）')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('模型参数灵敏度分析图.png', dpi=300)
        plt.close()
        print(f"  模型参数图已保存：{os.path.abspath('模型参数灵敏度分析图.png')}")

    # 主运行
    def run(self, excel_path: str, out_path: str = "女胎异常判定结果.xlsx", run_sensitivity: bool = True):
        print("=" * 70)
        print("          女胎异常判定分析流程")
        print("=" * 70)
        # 执行核心流程
        df = self.load_excel(excel_path)
        df = self.preprocess_data(df)
        self.analyze_feature_effectiveness(df)
        df_with_score = self.train_isolation_forest(df)
        df_with_risk = self.classify_risk_level(df_with_score)
        self.validate_model(df_with_risk)
        self.save_final_results(out_path)

        # 灵敏度分析
        if run_sensitivity:
            self.analyze_feature_sensitivity()
            self.analyze_threshold_sensitivity()
            self.analyze_model_param_sensitivity()

        # 总结
        print("\n" + "=" * 70)
        print("          分析完成！")
        print("=" * 70)
        print("核心结果：")
        print(f"  - 原始样本数：{len(self.raw_data)}个")
        print(f"  - 有效样本数：{len(self.processed_data)}个")
        # 风险分布
        risk_count = df_with_risk['风险等级'].value_counts()
        risk_pct = (df_with_risk['风险等级'].value_counts(normalize=True) * 100).round(1)
        for level in ['高风险', '中风险', '低风险']:
            print(f"    • {level}：{risk_count.get(level, 0)}个（{risk_pct.get(level, 0.0)}%）")

        print("\n生成文件：")
        print(f"  1. 判定结果：{os.path.abspath(out_path)}")
        print(f"  2. 特征分布：{os.path.abspath('特征分布分析图.png')}")
        if run_sensitivity:
            print(f"  3. 特征灵敏度：{os.path.abspath('特征灵敏度分析图.png')}")
            print(f"  4. 阈值灵敏度：{os.path.abspath('高风险阈值灵敏度分析图.png')}、{os.path.abspath('中风险阈值灵敏度分析图.png')}")
            print(f"  5. 模型参数灵敏度：{os.path.abspath('模型参数灵敏度分析图.png')}")

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        excel_path = sys.argv[1]
    else:
        excel_path = 'nvtai.xlsx'

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = '女胎异常判定结果.xlsx'

    analyzer = FetalAbnormalityAnalyzer()
    try:
        analyzer.run(excel_path, out_path, run_sensitivity=True)
    except Exception as e:
        print(f"分析出错：{str(e)}")
        sys.exit(1)