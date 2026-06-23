```python
import pandas as pd
import numpy as np

RAW_PATH = "house_Prediction.csv" 

COLUMN_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

MISSING_PLACEHOLDERS = ["", " ", "NA", "N/A", "na", "n/a", "null",
                         "NULL", "?", "-", "--", "None", "none", "nan"]

df = pd.read_csv(
    RAW_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES,
    engine="python",
    na_values=MISSING_PLACEHOLDERS,
    skip_blank_lines=True,
)

print(f"Shape: {df.shape}")
df.head()
```

    Shape: (506, 14)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
df.describe()
```

    CRIM       float64
    ZN         float64
    INDUS      float64
    CHAS         int64
    NOX        float64
    RM         float64
    AGE        float64
    DIS        float64
    RAD          int64
    TAX        float64
    PTRATIO    float64
    B          float64
    LSTAT      float64
    MEDV       float64
    dtype: object
    
    Missing values per column:
    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    MEDV       0
    dtype: int64
    
    Duplicate rows: 0
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = [c.strip().upper() for c in df.columns]

obj_cols = df.select_dtypes(include="object").columns.tolist()
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()

print(f"Text columns standardized: {obj_cols or 'none'}")
```

    Text columns standardized: none
    


```python
for col in df.columns:
    n_missing = df[col].isna().sum()
    if n_missing == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed '{col}' ({n_missing} missing) with median = {median_val:.3f}")
    else:
        mode_val = df[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed '{col}' ({n_missing} missing) with mode = '{mode_val}'")

print(f"\nRemaining missing values: {df.isna().sum().sum()}")
```

    
    Remaining missing values: 0
    


```python
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - df.shape[0]} duplicate rows. New shape: {df.shape}")
```

    Removed 0 duplicate rows. New shape: (506, 14)
    


```python
# Binary indicator column
df["CHAS"] = pd.to_numeric(df["CHAS"], errors="coerce").round().astype("Int64")
bad_chas = ~df["CHAS"].isin([0, 1])
print(f"CHAS values outside {{0,1}}: {bad_chas.sum()}")

# Discrete index column
df["RAD"] = pd.to_numeric(df["RAD"], errors="coerce")
print(f"RAD non-integer values: {(df['RAD'] % 1 != 0).sum()}")

# Force every other column to numeric, catching stray text/format issues
for col in df.columns:
    if col == "CHAS":
        continue
    coerced = pd.to_numeric(df[col], errors="coerce")
    newly_invalid = coerced.isna().sum() - df[col].isna().sum()
    if newly_invalid > 0:
        print(f"'{col}': {newly_invalid} non-numeric entries coerced to NaN")
    df[col] = coerced

# Fill any NaNs introduced by coercion
for col in df.columns[df.isna().any()]:
    med = df[col].median()
    df[col] = df[col].fillna(med)
    print(f"Post-standardization fill: '{col}' -> median {med:.3f}")
```

    CHAS values outside {0,1}: 0
    RAD non-integer values: 0
    


```python
print(f"Final shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

df.to_csv("cleaned_house_data.csv", index=False)
df.head()
```

    Final shape: (506, 14)
    Missing values: 0
    Duplicate rows: 0
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("cleaned_house_data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary = pd.DataFrame({
    "mean": df.mean(numeric_only=True),
    "median": df.median(numeric_only=True),
    "mode": df.mode(numeric_only=True).iloc[0],
    "std": df.std(numeric_only=True),
    "min": df.min(numeric_only=True),
    "max": df.max(numeric_only=True),
})
summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mode</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>3.613524</td>
      <td>0.25651</td>
      <td>0.01501</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>11.363636</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>11.136779</td>
      <td>9.69000</td>
      <td>18.10000</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>0.069170</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>0.554695</td>
      <td>0.53800</td>
      <td>0.53800</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>6.284634</td>
      <td>6.20850</td>
      <td>5.71300</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>68.574901</td>
      <td>77.50000</td>
      <td>100.00000</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>3.795043</td>
      <td>3.20745</td>
      <td>3.49520</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>9.549407</td>
      <td>5.00000</td>
      <td>24.00000</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>408.237154</td>
      <td>330.00000</td>
      <td>666.00000</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>18.455534</td>
      <td>19.05000</td>
      <td>20.20000</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>356.674032</td>
      <td>391.44000</td>
      <td>396.90000</td>
      <td>91.294864</td>
      <td>0.32000</td>
      <td>396.9000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>12.653063</td>
      <td>11.36000</td>
      <td>6.36000</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>37.9700</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>22.532806</td>
      <td>21.20000</td>
      <td>50.00000</td>
      <td>9.197104</td>
      <td>5.00000</td>
      <td>50.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>506.0</td>
      <td>3.613524</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>0.082045</td>
      <td>0.25651</td>
      <td>3.677083</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>506.0</td>
      <td>11.363636</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>12.500000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>506.0</td>
      <td>11.136779</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>5.190000</td>
      <td>9.69000</td>
      <td>18.100000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>506.0</td>
      <td>0.069170</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>506.0</td>
      <td>0.554695</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.449000</td>
      <td>0.53800</td>
      <td>0.624000</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>506.0</td>
      <td>6.284634</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>5.885500</td>
      <td>6.20850</td>
      <td>6.623500</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>506.0</td>
      <td>68.574901</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>45.025000</td>
      <td>77.50000</td>
      <td>94.075000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>506.0</td>
      <td>3.795043</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>2.100175</td>
      <td>3.20745</td>
      <td>5.188425</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>506.0</td>
      <td>9.549407</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>5.00000</td>
      <td>24.000000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>506.0</td>
      <td>408.237154</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>279.000000</td>
      <td>330.00000</td>
      <td>666.000000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>506.0</td>
      <td>18.455534</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>17.400000</td>
      <td>19.05000</td>
      <td>20.200000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>506.0</td>
      <td>356.674032</td>
      <td>91.294864</td>
      <td>0.32000</td>
      <td>375.377500</td>
      <td>391.44000</td>
      <td>396.225000</td>
      <td>396.9000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>506.0</td>
      <td>12.653063</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>6.950000</td>
      <td>11.36000</td>
      <td>16.955000</td>
      <td>37.9700</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>506.0</td>
      <td>22.532806</td>
      <td>9.197104</td>
      <td>5.00000</td>
      <td>17.025000</td>
      <td>21.20000</td>
      <td>25.000000</td>
      <td>50.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.hist(figsize=(16, 12), bins=30, edgecolor="black")
plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](output_10_0.png)
    



```python
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
    axes[i].set_title(col)

# hide unused subplots
for j in range(len(df.columns), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    



```python
plt.figure(figsize=(6, 5))
sns.boxplot(x="CHAS", y="MEDV", data=df, palette="Set2")
plt.title("House Value (MEDV) by Charles River Proximity (CHAS)")
plt.xlabel("Bounds Charles River (0 = No, 1 = Yes)")
plt.ylabel("Median Home Value ($1000s)")
plt.show()
```

    C:\Users\FBDA22-007\AppData\Local\Temp\ipykernel_28208\3319965751.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x="CHAS", y="MEDV", data=df, palette="Set2")
    


    
![png](output_12_1.png)
    



```python
corr = df.corr(numeric_only=True)
corr.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>1.00</td>
      <td>-0.20</td>
      <td>0.41</td>
      <td>-0.06</td>
      <td>0.42</td>
      <td>-0.22</td>
      <td>0.35</td>
      <td>-0.38</td>
      <td>0.63</td>
      <td>0.58</td>
      <td>0.29</td>
      <td>-0.39</td>
      <td>0.46</td>
      <td>-0.39</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>-0.20</td>
      <td>1.00</td>
      <td>-0.53</td>
      <td>-0.04</td>
      <td>-0.52</td>
      <td>0.31</td>
      <td>-0.57</td>
      <td>0.66</td>
      <td>-0.31</td>
      <td>-0.31</td>
      <td>-0.39</td>
      <td>0.18</td>
      <td>-0.41</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>0.41</td>
      <td>-0.53</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>0.76</td>
      <td>-0.39</td>
      <td>0.64</td>
      <td>-0.71</td>
      <td>0.60</td>
      <td>0.72</td>
      <td>0.38</td>
      <td>-0.36</td>
      <td>0.60</td>
      <td>-0.48</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>-0.06</td>
      <td>-0.04</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>-0.10</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>-0.12</td>
      <td>0.05</td>
      <td>-0.05</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>0.42</td>
      <td>-0.52</td>
      <td>0.76</td>
      <td>0.09</td>
      <td>1.00</td>
      <td>-0.30</td>
      <td>0.73</td>
      <td>-0.77</td>
      <td>0.61</td>
      <td>0.67</td>
      <td>0.19</td>
      <td>-0.38</td>
      <td>0.59</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>-0.22</td>
      <td>0.31</td>
      <td>-0.39</td>
      <td>0.09</td>
      <td>-0.30</td>
      <td>1.00</td>
      <td>-0.24</td>
      <td>0.21</td>
      <td>-0.21</td>
      <td>-0.29</td>
      <td>-0.36</td>
      <td>0.13</td>
      <td>-0.61</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.35</td>
      <td>-0.57</td>
      <td>0.64</td>
      <td>0.09</td>
      <td>0.73</td>
      <td>-0.24</td>
      <td>1.00</td>
      <td>-0.75</td>
      <td>0.46</td>
      <td>0.51</td>
      <td>0.26</td>
      <td>-0.27</td>
      <td>0.60</td>
      <td>-0.38</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-0.38</td>
      <td>0.66</td>
      <td>-0.71</td>
      <td>-0.10</td>
      <td>-0.77</td>
      <td>0.21</td>
      <td>-0.75</td>
      <td>1.00</td>
      <td>-0.49</td>
      <td>-0.53</td>
      <td>-0.23</td>
      <td>0.29</td>
      <td>-0.50</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.63</td>
      <td>-0.31</td>
      <td>0.60</td>
      <td>-0.01</td>
      <td>0.61</td>
      <td>-0.21</td>
      <td>0.46</td>
      <td>-0.49</td>
      <td>1.00</td>
      <td>0.91</td>
      <td>0.46</td>
      <td>-0.44</td>
      <td>0.49</td>
      <td>-0.38</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>0.58</td>
      <td>-0.31</td>
      <td>0.72</td>
      <td>-0.04</td>
      <td>0.67</td>
      <td>-0.29</td>
      <td>0.51</td>
      <td>-0.53</td>
      <td>0.91</td>
      <td>1.00</td>
      <td>0.46</td>
      <td>-0.44</td>
      <td>0.54</td>
      <td>-0.47</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>0.29</td>
      <td>-0.39</td>
      <td>0.38</td>
      <td>-0.12</td>
      <td>0.19</td>
      <td>-0.36</td>
      <td>0.26</td>
      <td>-0.23</td>
      <td>0.46</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>-0.18</td>
      <td>0.37</td>
      <td>-0.51</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.39</td>
      <td>0.18</td>
      <td>-0.36</td>
      <td>0.05</td>
      <td>-0.38</td>
      <td>0.13</td>
      <td>-0.27</td>
      <td>0.29</td>
      <td>-0.44</td>
      <td>-0.44</td>
      <td>-0.18</td>
      <td>1.00</td>
      <td>-0.37</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>0.46</td>
      <td>-0.41</td>
      <td>0.60</td>
      <td>-0.05</td>
      <td>0.59</td>
      <td>-0.61</td>
      <td>0.60</td>
      <td>-0.50</td>
      <td>0.49</td>
      <td>0.54</td>
      <td>0.37</td>
      <td>-0.37</td>
      <td>1.00</td>
      <td>-0.74</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>-0.39</td>
      <td>0.36</td>
      <td>-0.48</td>
      <td>0.18</td>
      <td>-0.43</td>
      <td>0.70</td>
      <td>-0.38</td>
      <td>0.25</td>
      <td>-0.38</td>
      <td>-0.47</td>
      <td>-0.51</td>
      <td>0.33</td>
      <td>-0.74</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
```


    
![png](output_14_0.png)
    



```python
target_corr = corr["MEDV"].drop("MEDV").sort_values(key=abs, ascending=False)
print("Correlation with MEDV (sorted by strength):")
print(target_corr)

plt.figure(figsize=(8, 6))
target_corr.plot(kind="barh", color=target_corr.apply(lambda x: "crimson" if x < 0 else "steelblue"))
plt.title("Feature Correlation with MEDV")
plt.xlabel("Correlation coefficient")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
```

    Correlation with MEDV (sorted by strength):
    LSTAT     -0.737663
    RM         0.695360
    PTRATIO   -0.507787
    INDUS     -0.483725
    TAX       -0.468536
    NOX       -0.427321
    CRIM      -0.388305
    RAD       -0.381626
    AGE       -0.376955
    ZN         0.360445
    B          0.333461
    DIS        0.249929
    CHAS       0.175260
    Name: MEDV, dtype: float64
    


    
![png](output_15_1.png)
    



```python
top_features = target_corr.abs().sort_values(ascending=False).head(4).index

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    sns.scatterplot(x=df[feat], y=df["MEDV"], ax=axes[i], alpha=0.6)
    sns.regplot(x=df[feat], y=df["MEDV"], ax=axes[i], scatter=False, color="red")
    axes[i].set_title(f"{feat} vs MEDV (r = {corr.loc[feat, 'MEDV']:.2f})")

plt.tight_layout()
plt.show()
```


    
![png](output_16_0.png)
    



```python
key_cols = ["RM", "LSTAT", "PTRATIO", "MEDV"]
sns.pairplot(df[key_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()
```


    
![png](output_17_0.png)
    



```python

```
