[原始网页](https://physionet.org/content/challenge-2019/1.0.0/)

# Challenge Data

Data used in the competition is sourced from ICU patients in three separate hospital systems. Data from two hospital systems will be publicly available; however, one data set will be censored and used for scoring. The data for each patient will be contained within a single pipe-delimited text file. Each file will have the same header and each row will represent a single hour's worth of data. Available patient co-variates consist of Demographics, Vital Signs, and Laboratory values, which are defined in the tables below.

比赛中使用的数据来自三个独立医院系统中的ICU患者。 来自两个医院系统的数据将公开提供； 但是，一个数据集将被检查并用于评分。 每个患者的数据将包含在单个竖线分隔文本文件中。 每个文件将具有相同的标题，并且每一行将代表一个小时的数据量。 可用的患者协变量由人口统计，生命体征和实验室值组成，在下表中定义。

---

The following time points are defined for each patient:

为每个患者定义以下时间点：

## t_suspicion

1. Clinical suspicion of infection identified as the earlier timestamp of IV antibiotics and blood cultures within a specified duration.

    临床怀疑感染是指在指定的时间内IV抗生素和血液培养的较早时间戳记。

2. If antibiotics were given first, then the cultures must have been obtained within 24 hours. If cultures were obtained first, then antibiotic must have been subsequently ordered within 72 hours.

    如果首先使用抗生素，则必须在24小时内获得培养物。 如果首先获得培养物，则随后必须在72小时内订购抗生素。

3. Antibiotics must have been administered for at least 72 consecutive hours to be considered.

    必须连续至少72小时服用抗生素才能考虑使用。

## t_SOFA

The occurrence of end organ damage as identified by a two-point deterioration in SOFA score within a 24-hour period.

通过在24小时内SOFA得分出现两点恶化来确定最终器官损伤的发生。

## t_sepsis

The onset time of sepsis is the earlier of tsuspicion and tSOFA as long as tSOFA occurs no more than 24 hours before or 12 hours after tsuspicion; otherwise, the patient is not marked as a sepsis patient. Specifically, if tsuspicion − 24 ≤ tSOFA ≤ tsuspicion + 12, then tsepsis = min(tsuspicion, tSOFA).

脓毒症的发作时间只要是tsuspicion和tSOFA中的较早时间，只要tSOFA发生在tsuspicion之前或之后的24小时内即可。 否则，该患者不会被标记为败血症患者。 具体来说，如果tsuspicion - 24 ≤ tSOFA ≤ tsuspicion + 12，则 tsepsis = min（tsuspicion，tSOFA）。

---

## Table 1: Columns in each training data file.

### Vital signs 生命体征 (columns 1-8)

| | | |
| --- | --- | --- |
| HR | Heart rate (beats per minute) | 心率 |
| O2Sat | Pulse oximetry (%) | 脉搏血氧 |
| Temp | Temperature (Deg C) | 体温 |
| SBP | Systolic BP (mm Hg) | 收缩压 |
| MAP | Mean arterial pressure (mm Hg) | 平均动脉压 |
| DBP | Diastolic BP (mm Hg) | 舒张压 |
| Resp | Respiration rate (breaths per minute) | 呼吸频率 |
| EtCO2 | End tidal carbon dioxide (mm Hg) | 呼出二氧化碳 |
| | | |

### Laboratory values 实验室值 (columns 9-34)

| | | |
| --- | --- | --- |
| BaseExcess | Measure of excess bicarbonate (mmol/L) | 碳酸氢根过量的量度 |
| HCO3 | Bicarbonate (mmol/L) | 碳酸氢盐 |
| FiO2 | Fraction of inspired oxygen (%) | 吸入氧气浓度 |
| pH | N/A | pH值 |
| PaCO2 | Partial pressure of carbon dioxide from arterial blood (mm Hg) | 动脉血的二氧化碳分压 |
| SaO2 | Oxygen saturation from arterial blood (%) | 动脉血的氧饱和度 |
| AST | Aspartate transaminase (IU/L) | 天冬氨酸转氨酶 |
| BUN | Blood urea nitrogen (mg/dL) | 血尿素氮 |
| Alkalinephos | Alkaline phosphatase (IU/L) | 碱性磷酸酶 |
| Calcium | (mg/dL) | 钙 |
| Chloride | (mmol/L) | 氯 |
| Creatinine | (mg/dL) | 肌酐 |
| Bilirubin_direct | Bilirubin direct (mg/dL) | 胆红素 |
| Glucose | Serum glucose (mg/dL) | 血清葡萄糖 |
| Lactate | Lactic acid (mg/dL) | 乳酸 |
| Magnesium | (mmol/dL) | 镁 |
| Phosphate | (mg/dL) | 磷酸盐 |
| Potassium | (mmol/L) | 钾盐 |
| Bilirubin_total | Total bilirubin (mg/dL) | 总胆红素 |
| TroponinI | Troponin I (ng/mL) | 肌钙蛋白I |
| Hct | Hematocrit (%) | 血细胞比容  |
| Hgb | Hemoglobin (g/dL) | 血红蛋白 |
| PTT | partial thromboplastin time (seconds) | 凝血活酶时间 |
| WBC | Leukocyte count (count*10^3/µL) | 白细胞计数 |
| Fibrinogen | (mg/dL) | 纤维蛋白原 |
| Platelets | (count*10^3/µL) | 血小板 |
| | | |

### Demographics 人口统计信息 (columns 35-40)

| | | |
| --- | --- | --- |
| Age | Years (100 for patients 90 or above) | 年龄（90岁及以上记为100岁）
| Gender | Female (0) or Male (1) | 性别（女性为0，男性为1）
| Unit1 | Administrative identifier for ICU unit (MICU) | ICU单位的管理标识符(MICU) |
| Unit2 | Administrative identifier for ICU unit (SICU) | ICU单位的管理标识符(SICU) |
| HospAdmTime | Hours between hospital admit and ICU admit | 住院到ICU之间的时间 |
| ICULOS | ICU length-of-stay (hours since ICU admit) | ICU住院时间（自ICU入院以来的小时数） |
| | | |

### Outcome 结果 (column 41)

||||
| --- | --- | --- |
| SepsisLabel | For sepsis patients, `SepsisLabel` is 1 if t ≥ tsepsis − 6 and 0 if t < tsepsis − 6. For non-sepsis patients, `SepsisLabel` is 0. | 对于败血症患者，如果t ≥ tsepsis − 6，则“ SepsisLabel”为1；如果t < tsepsis − 6，则为0。对于非败血症患者，“ SepsisLabel”为0。 |
||||

# Data Description

The Challenge data repository contains one file per subject (e.g., for the training data). `training/p00101.psv`

挑战数据存储库每个主题包含一个文件（例如，用于训练数据）。

## Accessing the Data

* [Click here](https://archive.physionet.org/pnw/challenge-2019-request-access) to download the complete training database (42 MB), consisting of two parts: training set A (20,336 subjects) and B (20,000 subjects).

    [单击此处](https://archive.physionet.org/pnw/challenge-2019-request-access)，下载完整的培训数据库（42 MB），其中包括两个部分：培训A组（20,336个科目）和B组（20,000个科目）。

Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time). The table is formatted in the following way:

每个训练数据文件都提供了一个表格，其中包含随着时间的推移的测量结果。 表格的每一列提供了一段时间内的一系列测量结果（例如，数小时内的心率），其中该列的标题描述了该测量结果。 表格的每一行同时提供了一组测量值（例如，同时显示了心率和氧气水平）。 该表的格式如下：

```
HR |O2Sat|Temp|...|HospAdmTime|ICULOS|SepsisLabel
NaN|  NaN| NaN|...|        -50|     1|          0
 86|   98| NaN|...|        -50|     2|          0
 75|  NaN| NaN|...|        -50|     3|          1
 99|  100|35.5|...|        -50|     4|          1
```

There are 40 time-dependent variables , , ..., , which are described here. The final column, , indicates the onset of sepsis according to the Sepsis-3 definition, where indicates sepsis and indicates no sepsis. Entries of (not a number) indicate that there was no recorded measurement of a variable at the time interval. `HR` `O2Sat` `Temp` `HospAdmTime` `SepsisLabel` `1` `0` `NaN`

这里有40个与时间相关的变量，，...，。 最后一列表示根据败血症3定义的败血症发作，其中指示败血症，不指示败血症。 项（非数字）表示在该时间间隔内没有记录到的变量测量值。

Note: spaces were added to this example to improve readability. They will not be present in the data files.

注意：此示例中添加了空格以提高可读性。 它们将不会出现在数据文件中。
