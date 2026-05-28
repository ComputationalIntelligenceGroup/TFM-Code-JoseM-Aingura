from pathlib import Path
import pandas as pd

# Configure pandas display options to show all digits without scientific notation
pd.set_option('display.float_format', lambda x: f'{x:.10f}' if abs(x) < 1e10 else str(x))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Use the script's directory as reference point
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data2"

# Load datasets from data2
df_features = pd.read_csv(DATA_DIR / "MT002_ENSAYO_3_3070_IE08_FEATURES.csv", sep=';')
df_kpi = pd.read_csv(DATA_DIR / "MT002_ENSAYO3_3070_IE08_KPI.csv", sep=';')

# Remove first 57826 rows from df_features
df_features = df_features.iloc[57826:]

# Make df_features and df_kpi have the same number of rows
df_features = df_features.iloc[:len(df_kpi)]

# Add likelihood column from df_kpi to df_features
df_features['likelihood'] = df_kpi['likelihood'].values

# Display dataframe info
print("Features DataFrame:")
print(f"  Shape: {df_features.shape}")
print(f"  Columns: {list(df_features.columns)}")
print("\nFirst row of df_features:")
print(df_features.iloc[0])
print("\nLast row of df_features:")
print(df_features.iloc[-1])
print("\nKPI DataFrame:")
print(f"  Shape: {df_kpi.shape}")
print(f"  Columns: {list(df_kpi.columns)}")
print("\nFirst row of df_kpi:")
print(df_kpi.iloc[0])
print("\nLast row of df_kpi:")
print(df_kpi.iloc[-1])

# Save df_features to data2 folder
df_features.to_csv(DATA_DIR / "MT002_ENSAYO_3_NEW_FEATURES.csv", sep=';', index=False)
print("\n✓ df_features saved as MT002_ENSAYO_3_NEW_FEATURES.csv")


