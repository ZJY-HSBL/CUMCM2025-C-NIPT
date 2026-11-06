# 问题一
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, rankdata, pearsonr
from statsmodels.formula.api import mixedlm
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False




def WeekAndDay_to_Week(Week):
    # 将字符串转换为小写，统一处理大小写问题
    Week = Week.lower()

    # 查找 'w' 的位置，分离周数和天数
    W_Index = Week.find('w')

    if W_Index == -1:
        print(f"存在无法识别的孕周格式{Week}")
        return -1

    # 提取周数和天数
    try:
        W = int(Week[:W_Index])  # 周数部分
        D = int(Week[W_Index + 1:]) if len(Week) > W_Index + 1 else 0  # 天数部分，默认0
    except ValueError:
        print(f"存在无法识别的孕周格式{Week}")
        return -1

    # 将孕周转换为小数
    WEEK = W + D / 7
    return WEEK



def CheckBMI(Data):
    NewData=[]
    for i in Data:
        if i < 20:
            print(f"BMI应该大于20")
        else:
            NewData.append(i)
    return NewData


def GleanData_Question_1():
    data = pd.read_excel("a.xlsx")

    # 逐行处理检测孕周
    data["孕周"] = data["检测孕周"].apply(WeekAndDay_to_Week)

    # 检查孕周是否转换成功
    if data["孕周"].isna().sum() > 0:
        print("孕周格式转换失败")
        return -1

    # 处理BMI
    data["BMI"] = CheckBMI(data["孕妇BMI"])

    # 处理Y染色体浓度
    Y_C = data["Y染色体浓度"]
    data["Y染色体浓度"] = [i if i >= 0 else None for i in Y_C]

    # 保存清洗后的数据到新的 Excel 文件
    data.to_excel("a_1.xlsx", index=False)

    return data


def VisionData_question_1_1(DF):
    """
    将数据孕周和BMI离散图呈现分析数据间的关系
    """

    # 绘制原始数据的散点图
    plt.scatter(DF['孕周'], DF['BMI'], color='blue')
    plt.xlabel('Weeks')
    plt.ylabel('BMI')
    plt.title('孕周数与BMI的散点图')
    plt.show()


def VisionData_question_1_2(DF, choose=1):
    """
    将数据孕周和BMI 与 Y的浓度 离散图呈现分析数据间的关系
    """
    if choose == 1:
        # 绘制标准化数据的散点图
        plt.scatter(DF['孕周'], DF['Y染色体浓度'], color='blue')
        plt.xlabel('Weeks')
        plt.ylabel('Y染色体浓度')
        plt.title('孕周数与Y染色体浓度的散点图')
        plt.show()
    else:
        plt.scatter(DF['BMI'], DF['Y染色体浓度'], color='blue')
        plt.xlabel('BMI')
        plt.ylabel('Y染色体浓度')
        plt.title('BMI与Y染色体浓度的散点图')
        plt.show()



def FindRelation_BMI(X_Poly, y):
    if isinstance(X_Poly, (pd.DataFrame, pd.Series)):
        X_Poly = X_Poly.values  # 转换为NumPy数组
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values  # 转换为NumPy数组


    if X_Poly.ndim == 1:
        X_Poly = X_Poly.reshape(-1, 1)


    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly_transformed = poly.fit_transform(X_Poly)
    feature_names = poly.get_feature_names_out(['BMI'])  # 获取特征名称（便于解释系数）

    model_poly = LinearRegression()
    model_poly.fit(X_poly_transformed, y)

    # 预测
    y_Pred = model_poly.predict(X_poly_transformed)

    r2 = r2_score(y, y_Pred)  # 决定系数（解释力）
    mse = mean_squared_error(y, y_Pred)  # 均方误差（误差平方的均值）
    rmse = np.sqrt(mse)  # 均方根误差（误差的平均量级）
    mae = mean_absolute_error(y, y_Pred)  # 平均绝对误差（误差的平均绝对值）

    # 输出评价指标
    print("模型性能评价")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}\n")

    print("多项式回归方程系数")
    for name, coef in zip(feature_names, model_poly.coef_):
        if name == '1':
            print(f"常数项 (β0): {model_poly.intercept_:.4f}")
        else:
            print(f"{name} 系数 (β{name[-1]}): {coef:.4f}")

    sorted_indices = np.argsort(X_Poly.ravel())
    X_sorted = X_Poly.ravel()[sorted_indices]
    y_Pred_sorted = y_Pred[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_Poly, y, color='blue', alpha=0.5, label='实际数据')
    plt.plot(X_sorted, y_Pred_sorted, color='red', linewidth=2, label='二次多项式拟合')
    plt.xlabel('标准化BMI')
    plt.ylabel('Y染色体浓度')
    plt.title(f'BMI与Y染色体浓度的关系（R² = {r2:.4f}）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return {
        'model': model_poly,
        'poly_transformer': poly,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'coefficients': dict(zip(feature_names, [model_poly.intercept_] + list(model_poly.coef_)[1:]))
    }


def FindRelation_Week(DF,DF_Code, y):

    import statsmodels.api as sm

    X = sm.add_constant(DF[['孕周', 'BMI'] + list(DF_Code.columns)])  # 特征矩阵

    X = X.apply(pd.to_numeric, errors='coerce')

    print("特征数据类型：")
    print(X.dtypes)
    print("\n目标变量数据类型：")
    print(y.dtypes)

    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print("\n发现非数值特征列：", non_numeric_cols.tolist())

    print("\n缺失值统计：")
    print(X.isnull().sum())

    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print("转换后的数据类型：")
    print(X.dtypes)

    model_fe = sm.OLS(y, X).fit()

    print(model_fe.summary())




def question_1():
    # 数据清洗
    # GleanData_Question_1()
    # 数据标准化
    Data = pd.read_excel("a_1.xlsx")
    DF = Data[['孕妇代码', '孕周', 'BMI', 'Y染色体浓度']]
    # 对孕周、BMI、Y染色体浓度进行标准化
    scaler = StandardScaler()
    DF[['孕周', 'BMI', 'Y染色体浓度']] = scaler.fit_transform(DF[['孕周', 'BMI', 'Y染色体浓度']])
    print(DF)


    #DF.to_excel("a_1_1.xlsx", index=False)

    # 分析孕周和BMI的关系
    VisionData_question_1_1(DF) # 散点图显示出没有明显的线性或非线性关系
    # 分析Y染色体浓度和BMI的关系
    VisionData_question_1_2(DF, choose=1)
    # 分析Y染色体浓度和孕周的关系
    VisionData_question_1_2(DF, choose=2)

    # 提取特征和目标变量
    X_Poly = DF[['BMI', '孕周']]  # 使用BMI来预测Y染色体浓度
    y = DF['Y染色体浓度']
    FindRelation_BMI(X_Poly,y)
    # FindRelation_Week(DF,DF_Code, y)

    correlation = DF[['BMI', '孕周']].corr()
    print(correlation)


    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    # 选择需要计算VIF的特征
    X = DF[['BMI', '孕周']]

    # 添加常数项（截距）
    X_const = add_constant(X)

    # 计算VIF
    vif = pd.DataFrame()
    vif["Variable"] = X_const.columns
    vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    print(vif)



def GleanData_Question_1():
    """
    对第一问需要的数据进行清洗和处理
    BMI: > 20
    Y染色体: > 0
    孕周格式统一: 周数 + 天数 格式转换为 小数周数
    """
    data = pd.read_excel("a.xlsx")

    # 逐行处理检测孕周
    data["孕周"] = data["检测孕周"].apply(data.WeekAndDay_to_Week)

    # 检查孕周是否转换成功
    if data["孕周"].isna().sum() > 0:
        print("孕周格式转换失败")
        return -1

    # 处理BMI
    data["BMI"] = data.CheckBMI(data["孕妇BMI"])

    # 处理Y染色体浓度
    Y_C = data["Y染色体浓度"]
    data["Y染色体浓度"] = [i if i >= 0 else None for i in Y_C]

    # 保存清洗后的数据到新的 Excel 文件
    #data.to_excel("a_1.xlsx", index=False)

    return data


# 偏相关矩阵
def pianjuzhen(shuju):
    cols = shuju.columns
    n = len(cols)
    pcm = pd.DataFrame(np.eye(n), index=cols, columns=cols)

    for i in range(n):
        for j in range(i+1, n):
            others = [k for k in range(n) if k not in (i, j)]
            if others:  # 如果有控制变量
                y_i = rankdata(shuju.iloc[:, i])
                y_j = rankdata(shuju.iloc[:, j])
                C   = np.column_stack([rankdata(shuju.iloc[:, k]) for k in others])

                # 残差法：回归去除控制变量
                C_aug = np.c_[np.ones(len(y_i)), C]
                beta_i = np.linalg.lstsq(C_aug, y_i, rcond=None)[0]
                beta_j = np.linalg.lstsq(C_aug, y_j, rcond=None)[0]
                r_i = y_i - C_aug @ beta_i
                r_j = y_j - C_aug @ beta_j

                r, _ = pearsonr(r_i, r_j)
            else:
                r, _ = spearmanr(shuju.iloc[:, i], shuju.iloc[:, j])

            pcm.iloc[i, j] = pcm.iloc[j, i] = r
    return pcm


# 秩回归
def zhihuigui(y, X):
    # 秩变换
    y_r = rankdata(y).astype(float)
    X_r = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]).astype(float)

    n, p = X_r.shape
    X_aug = np.c_[np.ones(n), X_r]  # 加截距
    beta = np.linalg.lstsq(X_aug, y_r, rcond=None)[0]

    # 拟合优度
    y_hat = X_aug @ beta
    resid = y_r - y_hat
    sse = resid @ resid
    sst = ((y_r - y_r.mean())**2).sum()
    r2 = 1 - sse/sst

    return {
        "beta": beta,
        "R2": r2,
        "y_hat": y_hat
    }





def vision_pianjuzhen(pianjuzhen):
    # 偏相关矩阵热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(pianjuzhen, annot=True, fmt=".3f", cmap="binary", square=True, cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    plt.show()


def vision_zhihuigui_1(y,res):
    # 实际值 预测值 散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=res['y_hat'], color="black", markers="o")
    plt.xlabel("实际值 (y)", fontsize=15)
    plt.ylabel("拟合值 (y_hat)", fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.show()


def vision_zhihuigui_2(y,res):
    # 拟合线与散点图
    plt.figure(figsize=(8, 6))

    # 保存regplot的返回值
    ax = sns.regplot(
        x=res['y_hat'],
        y=res['y_hat'] - y,
        lowess=True,
        scatter_kws={"color": "black", "label": "残差点", "alpha": 0.1, "marker": "o"},
        line_kws={"color": "black", "label": "LOWESS曲线"}
    )

    # 获取散点和线条对象
    dian = ax.collections[0]  # 散点
    xian = ax.lines[0]  # 线

    # 手动创建图例
    plt.legend(
        [dian, xian],
        ["残差点", "LOWESS曲线"],
        fontsize=12
    )

    plt.xlabel("拟合值", fontsize=15)
    plt.ylabel("残差", fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.show()


def vision_pinhuaquxian(shuju,A='孕周',B='MBI'):
    # 生成平滑曲线数据（按孕周）
    min_gw, max_gw = shuju[A].min(), shuju[A].max()
    gw_range = np.linspace(min_gw, max_gw, 100)
    mean_bmi = shuju[B].mean()  # 使用平均BMI值作为控制变量

    # 计算线性模型的平滑预测值
    linear_pred_smooth = (params_linear['Intercept'] +
                          params_linear[A] * gw_range +
                          params_linear[B] * mean_bmi)

    # 计算非线性模型的平滑预测值
    nonlinear_pred_smooth = (params_nonlinear['Intercept'] +
                             params_nonlinear[A] * gw_range +
                             params_nonlinear['GW_squared'] * (gw_range ** 2) +
                             params_nonlinear[B] * mean_bmi +
                             params_nonlinear['BMI_squared'] * (mean_bmi ** 2))

    # 绘制孕周与Y染色体浓度的平滑拟合曲线对比
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=shuju[A],
        y=shuju['Y染色体浓度'],
        label='原始数据',
        marker='o',  # 指定标记为圆圈
        edgecolor='black',  # 圆圈边缘颜色设为黑色
        facecolor='none',  # 圆圈内部设为透明
    )
    plt.plot(gw_range, linear_pred_smooth, 'k-', label='线性模型拟合', linewidth=2)
    plt.plot(gw_range, nonlinear_pred_smooth, 'k--', label='非线性模型拟合', linewidth=2)

    plt.xlabel(A, fontsize=15)
    plt.ylabel('Y染色体浓度', fontsize=15)
    plt.legend()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.show()


def vision_pinjia(x, r2_values, mse_values, width = 0.35,models = ['线性模型', '非线性模型']):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- 左轴：R² ---
    bars1 = ax1.bar(x - width / 2, r2_values, width,
                    label='决定系数',
                    facecolor='white',
                    edgecolor='black',
                    hatch='//')  # 斜线填充
    ax1.set_ylabel('决定系数')
    ax1.set_ylim(0, max(r2_values) + 0.1)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    # --- 右轴：MSE ---
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mse_values, width,
                    label='均方误差',
                    facecolor='white',
                    edgecolor='black',
                    hatch='..')  # 点纹填充
    ax2.set_ylabel('均方误差')
    ax2.set_ylim(0, max(mse_values) * 1.2)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    # --- X轴 ---
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    # 标题 & 图例
    fig.suptitle('模型评价指标对比', fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def lingmingdu_test(shuju, extreme_ratios, extreme_magnitudes):
    results = []

    # 循环测试
    for ratio in extreme_ratios:
        for mag in extreme_magnitudes:
            df_extreme = shuju.copy()

            # 随机选择一定比例的样本并修改数据
            idx = df_extreme.sample(frac=ratio, random_state=42).index
            df_extreme.loc[idx, 'BMI'] += mag
            df_extreme.loc[idx, '孕周'] += mag / 10  # 孕周幅度较小，按比例调整

            # 定义并拟合非线性混合效应模型
            model_nonlinear_ext = MixedLM.from_formula(
                "Y染色体浓度 ~ 孕周 + GW_squared + BMI + BMI_squared",
                groups=df_extreme["孕妇代码"], data=df_extreme
            )
            result_nonlinear_ext = model_nonlinear_ext.fit()

            # 获取模型拟合值
            y_pred_ext = result_nonlinear_ext.fittedvalues

            # 计算R²和MSE
            r2_ext = r2_score(df_extreme['Y染色体浓度'], y_pred_ext)
            mse_ext = mean_squared_error(df_extreme['Y染色体浓度'], y_pred_ext)

            results.append({
                "比例": ratio,
                "幅度": mag,
                "R²": r2_ext,
                "MSE": mse_ext
            })

    # 转为 DataFrame 并显示结果
    res_df = pd.DataFrame(results)
    return res_df


def vision_lingmingdu(res_df,extreme_ratios):
    # 绘制 R² 灵敏度曲线
    plt.figure(figsize=(8, 6))

    # 定义不同的标记形状列表
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, ratio in enumerate(extreme_ratios):
        subset = res_df[res_df["比例"] == ratio]
        # 从形状列表中循环获取形状，使用取模操作避免索引超出范围
        marker = markers[i % len(markers)]
        plt.plot(subset["幅度"], subset["R²"], marker=marker, color='black', label=f"数据占比 {int(ratio * 100)}%")
    plt.axhline(r2_base, color='black', linestyle='--', label='原始R*R')
    plt.xlabel("极端数据幅度", fontsize=15)
    plt.ylabel("R*R", fontsize=15)
    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 6))
    # 绘制 MSE 灵敏度曲线
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, ratio in enumerate(extreme_ratios):
        subset = res_df[res_df["比例"] == ratio]
        # 从形状列表中循环获取形状，使用取模操作避免索引超出范围
        marker = markers[i % len(markers)]
        plt.plot(subset["幅度"], subset["MSE"], marker=marker, color='black', label=f"数据占比 {int(ratio * 100)}%")
    plt.axhline(mse_base, color='black', linestyle='--', label='原始MSE')
    plt.title("MSE 灵敏度分析", fontsize=15)
    plt.xlabel("极端数据幅度", fontsize=15)
    plt.ylabel("MSE", fontsize=15)
    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    shuju = pd.read_excel('a_1_1.xlsx')

    y = np.array(shuju['Y染色体浓度'])
    x1 = np.array(shuju['孕周'])
    x2 = np.array(shuju['BMI'])

    shuju_df = pd.DataFrame({"y染色体": y, "孕周": x1, "BMI": x2})

    # Spearman相关系数矩阵
    juzhen = shuju_df.corr(method="spearman")
    print(juzhen)
    # 相关矩阵热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(juzhen, annot=True, fmt=".3f", cmap="binary", square=True, cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    plt.show()

    # 偏相关矩阵
    pianjuzhen = pianjuzhen(shuju_df)
    print(pianjuzhen)

    # 秩回归
    X = np.column_stack([x1, x2])
    res = zhihuigui(y, X)
    vision_zhihuigui_1(y,res)
    vision_zhihuigui_2(y,res)

    # 定义线性混合效应模型
    model_linear = mixedlm("Y染色体浓度 ~ 孕周 + BMI", shuju, groups=shuju["孕妇代码"])
    result_linear = model_linear.fit(reml=False)

    # 输出线性模型结果
    print("\n线性混合效应模型\n")
    print(result_linear.summary())

    # 评价
    y_true = shuju['Y染色体浓度'].values
    y_pred_linear = result_linear.fittedvalues.values

    r2_linear = r2_score(y_true, y_pred_linear)
    mse_linear = mean_squared_error(y_true, y_pred_linear)

    print("\n线性模型评价指标:\n")
    print(f"R²（决定系数）: {r2_linear:.4f}")
    print(f"MSE（均方误差）: {mse_linear:.4f}")

    # 提取线性模型系数并输出公式
    params_linear = result_linear.params
    linear_formula = f"Y染色体浓度 = {params_linear['Intercept']:.4f} + {params_linear['孕周']:.4f}×孕周 + {params_linear['BMI']:.4f}×BMI + (随机截距)"
    print("\n线性混合效应模型公式:\n")
    print(linear_formula)

    # 准备非线性模型数据
    shuju['GW_squared'] = shuju['孕周'] ** 2
    shuju['BMI_squared'] = shuju['BMI'] ** 2

    # 定义非线性混合效应模型
    model_nonlinear = mixedlm("Y染色体浓度 ~ 孕周 + GW_squared + BMI + BMI_squared",
                    shuju, groups=shuju["孕妇代码"])
    result_nonlinear = model_nonlinear.fit(reml=False)

    # 评价
    y_pred_nonlinear = result_nonlinear.fittedvalues.values
    r2_nonlinear = r2_score(y_true, y_pred_nonlinear)
    mse_nonlinear = mean_squared_error(y_true, y_pred_nonlinear)

    print("\n非线性混合效应模型结果\n")
    print(result_nonlinear.summary())

    print("\n非线性模型评价指标:\n")
    print(f"R²（决定系数）: {r2_nonlinear:.4f}")
    print(f"MSE（均方误差）: {mse_nonlinear:.4f}")

    # 提取非线性模型系数并输出公式
    params_nonlinear = result_nonlinear.params
    nonlinear_formula = (f"Y染色体浓度 = {params_nonlinear['Intercept']:.4f} + {params_nonlinear['孕周']:.4f}×孕周 + {params_nonlinear['GW_squared']:.4f}×孕周² + "
                        f"{params_nonlinear['BMI']:.4f}×BMI + {params_nonlinear['BMI_squared']:.4f}×BMI² + (随机截距)")
    print("\n非线性混合效应模型公式\n")
    print(nonlinear_formula)

    vision_pinhuaquxian(shuju, '孕周', 'BMI')
    vision_pinhuaquxian(shuju, 'BMI', '孕周')

    models = ['线性模型', '非线性模型']
    r2_values = [r2_linear, r2_nonlinear]
    mse_values = [mse_linear, mse_nonlinear]
    x = np.arange(len(models))

    vision_pinjia(x, r2_values, mse_values)

    print("\n灵敏度测试\n")
    y_pred = result_nonlinear.fittedvalues
    r2_base = r2_score(shuju['Y染色体浓度'], y_pred)
    mse_base = mean_squared_error(shuju['Y染色体浓度'], y_pred)

    print(f"原始 R²: {r2_base:.4f}, MSE: {mse_base:.4f}")

    # 定义极端比例和幅度
    extreme_ratios = [0.05, 0.1, 0.2, 0.3]  # 极端比例
    extreme_magnitudes = [20, 50, 80]  # 极端幅度

    shuju_test=lingmingdu_test(shuju, extreme_ratios, extreme_magnitudes)
    print(shuju_test)

    vision_lingmingdu(shuju_test, extreme_ratios)