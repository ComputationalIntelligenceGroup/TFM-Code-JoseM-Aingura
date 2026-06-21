import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"..\data_ascensores\Features_Ascensores.csv"

SEP = ","
TIME_COL = "timestamp"
TIMESTAMP_UNIT = None  # timestamps are ISO8601 strings with Z timezone

GAP = pd.Timedelta(days=1)     # split whenever gap > 1 day
COLUMNS_TO_PLOT = None         # None = all columns except timestamp
DOWNSAMPLE_EVERY_N = 1         # set to 5/10 if slow


def plot_multiseries(df, cols, title):
    if df.empty:
        return
    if DOWNSAMPLE_EVERY_N > 1:
        df = df.iloc[::DOWNSAMPLE_EVERY_N]

    n = len(cols)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, sharex=True,
        figsize=(14, max(2.2 * n, 6)),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    step = 10000
    vline_positions = df.index[::step]

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col].to_numpy(), linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        for x in vline_positions:
            ax.axvline(x=x, color='red', linestyle='-', linewidth=0.8, alpha=0.7)

    axes[-1].set_xlabel("time")
    fig.suptitle(title, y=1.01)
    plt.show()


# --- load ---
df = pd.read_csv(CSV_PATH, sep=SEP)
print(df.columns.tolist())   # debug: check real column names

# optional cleanup in case headers have spaces/BOM
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
print(df.columns.tolist())   # debug: cleaned column names

if TIME_COL not in df.columns:
    raise ValueError(f"Expected time column '{TIME_COL}' not found in CSV")

# parse timestamps

df["time"] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.loc[df["time"].notna()].copy()
df = df.set_index("time").sort_index()

cols = [c for c in df.columns if c != TIME_COL]
if COLUMNS_TO_PLOT is not None:
    cols = [c for c in COLUMNS_TO_PLOT if c in df.columns]

# --- split into segments where time gap > 1 day ---
gaps = df.index.to_series().diff()
split_points = gaps[gaps > GAP].index

dfs = []
start_pos = 0
split_positions = [df.index.get_loc(t) for t in split_points]
for pos in split_positions + [len(df)]:
    dfs.append(df.iloc[start_pos:pos])
    start_pos = pos

print(f"Found {len(dfs)} segment(s) with GAP>{GAP}.")
for i, part in enumerate(dfs, 1):
    print(f"  Segment {i}: {part.index.min()} -> {part.index.max()} ({len(part):,} rows)")

for i, part in enumerate(dfs, 1):
    plot_multiseries(part, cols, f"Segment {i} (gap split > {GAP})")

if dfs:
    df1_clean = dfs[0].drop(columns=["time", TIME_COL], errors="ignore")
    df1_clean.to_csv("normal.csv", index=False, sep=SEP)
    print("Saved normal.csv")

    if len(dfs) > 1:
        df2_clean = dfs[1].drop(columns=["time", TIME_COL], errors="ignore")
        df2_clean.to_csv("anomalous.csv", index=False, sep=SEP)
        print("Saved anomalous.csv")
    else:
        print("Only one segment found; anomalous.csv was not created.")
else:
    print("No valid data segments found; nothing was saved.")
