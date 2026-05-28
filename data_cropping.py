
import pandas as pd





# LiNGAM (causal-learn

CI_TEST = "fisherz"

CSV_PATH = r"..\data\anomalous.csv" # CAMBIAR por normal.csv para ver datos normales 

SEP = ";"
TIME_COL = "timestamp"
TIMESTAMP_UNIT = "us"  

GAP = pd.Timedelta(days=1)     # split whenever gap > 1 day
COLUMNS_TO_PLOT = None         # None = all columns except timestamp
DOWNSAMPLE_EVERY_N = 1         # set to 5/10 if slow

chunk_size = 10000

df = pd.read_csv(CSV_PATH, sep=SEP)

df_chunk = df.iloc[130000: 140000]
prev_chunk = df.iloc[140000: 150000]



chunk_t_minus_1 = prev_chunk.reset_index(drop=True).add_suffix('_t-1')
chunk_t = df_chunk.reset_index(drop=True).add_suffix('_t')

df_concat = pd.concat([chunk_t_minus_1, chunk_t], axis=1)
df_concat.to_csv("tem_anomaly7.csv", index=False)