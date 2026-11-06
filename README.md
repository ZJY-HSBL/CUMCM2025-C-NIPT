# CUMCM2025-C-NIPT

**2025 年高教社杯全国大学生数学建模竞赛 C 题：NIPT 的时点选择与胎儿的异常判定**  
*2025 CUMCM, Problem C: NIPT Timing and Fetal Abnormality Assessment*

本仓库归档 2025 年参赛时提交的四个 Python 程序，同时提供事后审查记录。`src/original/` 是原始提交程序的逐字节副本，**不意味着模型已得到独立验证，也不意味着可用于医学决策**。已发现影响结果有效性的建模与验证问题，参见 [审查报告](docs/AUDIT.md)。

This repository preserves the four originally submitted Python scripts, alongside a post-submission technical audit. The original code has **not** been independently validated or cleared for clinical use. See [Audit report](docs/AUDIT.md).

## 目录 / Contents

| 文件 | 用途 |
|---|---|
| `src/original/question_1.py` | 孕周、BMI 与 Y 染色体浓度的相关性及混合效应模型 |
| `src/original/question_2.py` | BMI 聚类、PSO 时点估计与敏感性分析 |
| `src/original/question_3.py` | BMI 分层、多因素 Cox 模型与 Bootstrap |
| `src/original/question_4.py` | 女胎特征工程、孤立森林与风险分类 |
| `docs/AUDIT.md` | 审查发现、影响范围及建议修正方向 |
| `tools/run_submission.py` | 原样调用单题脚本的辅助运行器 |
| `tests/test_archive.py` | 源文件完整性和基本模型一致性审计测试 |

## 环境 / Environment

建议 Python 3.10–3.12，运行前安装：

```bash
python -m pip install -r requirements.txt
```

源程序使用了中文绘图标签，可能需要系统中文字体。运行原始代码前，请准备相应的本地数据文件，并检查 `docs/AUDIT.md`，尤其是问题二的变量标准化/4% 阈值不一致。文件路径和依赖仍保留原始设计，未将其包装成已经验证的复现流程。

| 题号 | 原始程序默认读取的文件 |
|---|---|
| 1 | `a_1_1.xlsx`（归一化/标准化结果） |
| 2 | `a_1.xlsx`（中间处理结果） |
| 3 | `附件.xlsx`（原始题目数据，工作表 `男胎检测数据`） |
| 4 | `nvtai.xlsx`（从题目女性数据单独整理的工作表） |

将对应数据文件放在本地 `private_data/` 目录后，可执行：

```bash
python tools/run_submission.py 1 --data-dir private_data
python tools/run_submission.py 2 --data-dir private_data
python tools/run_submission.py 3 --data-dir private_data
python tools/run_submission.py 4 --data-dir private_data --no-gui
```

**Data policy:** The original spreadsheets and case-level records are excluded from the repository, including the intermediary workbooks. Obtain the official CUMCM problem data from an authorized source. Do not upload individual-level records without confirming data-use permissions. The original submitted PDF is also excluded because the user requested a code repository; it remains in the original local submission archive.

## 结果使用声明 / Result validity

- 参赛论文报告的数字是当时的**提交结果**，不是此仓库独立复核后的再现结果。
- 问题二最佳时点及问题四异常分类评价存在关键方法学缺陷；仓库不会把这些结果包装为已验证的临床建议。
- Code archiving is not independent reproduction. Reported model metrics and timing estimates should not be interpreted as validated medical recommendations.

## License

尚未授权第三方重用；未附加开源许可证。For archival reference only unless the copyright holders specify otherwise.
