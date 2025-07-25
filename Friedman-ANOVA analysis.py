import pandas as pd
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load Excel file — replace 'your_path_here.xlsx' with your actual file path
df = pd.read_excel("your_path_here.xlsx")
df.rename(columns={"SER%": "SER", "Archivo de Audio": "Audio_File", "Umbral": "Threshold", "n_mels": "Mel_Bands", "hop_length": "Hop_Length"}, inplace=True)

print("=== Friedman and Repeated Measures ANOVA Results ===")

def run_friedman_pivot(df, index_cols, pivot_col, values_col, levels):
    pivot = df.pivot_table(index=index_cols, columns=pivot_col, values=values_col).dropna(subset=levels)
    if not pivot.empty:
        stat, p = friedmanchisquare(*(pivot[level] for level in levels))
        return stat, p, pivot.shape[0]
    return None, None, 0

def run_anova(df, formula):
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# --- Hop_Length ---
levels_hop = [110, 220, 330]
friedman_stat_hop, friedman_p_hop, pairs_hop = run_friedman_pivot(
    df,
    index_cols=["Audio_File", "Mel_Bands", "Threshold"],
    pivot_col="Hop_Length",
    values_col="SER",
    levels=levels_hop
)

anova_hop = df[df["Hop_Length"].isin(levels_hop)]
anova_table_hop = run_anova(anova_hop, 'SER ~ C(Hop_Length) + C(Audio_File)')

print("\n-- Hop_Length (110 vs 220 vs 330) --")
if friedman_stat_hop is not None:
    print(f"Friedman: Statistic={friedman_stat_hop:.4f}, p-value={friedman_p_hop:.4f}, pairs={pairs_hop}")
else:
    print("Friedman: Insufficient data")
print(anova_table_hop)

# --- Threshold ---
levels_threshold = [0.1, 0.2, 0.3, 0.4]
friedman_stat_threshold, friedman_p_threshold, pairs_threshold = run_friedman_pivot(
    df,
    index_cols=["Audio_File", "Mel_Bands", "Hop_Length"],
    pivot_col="Threshold",
    values_col="SER",
    levels=levels_threshold
)

anova_threshold = df[df["Threshold"].isin(levels_threshold)]
anova_table_threshold = run_anova(anova_threshold, 'SER ~ C(Threshold) + C(Audio_File)')

print("\n-- Threshold (0.1 vs 0.2 vs 0.3 vs 0.4) --")
if friedman_stat_threshold is not None:
    print(f"Friedman: Statistic={friedman_stat_threshold:.4f}, p-value={friedman_p_threshold:.4f}, pairs={pairs_threshold}")
else:
    print("Friedman: Insufficient data")
print(anova_table_threshold)

# --- Mel_Bands ---
levels_mel = [10, 20, 30, 40]
subset_mel = df[df["Mel_Bands"].isin(levels_mel)]
friedman_stat_mel, friedman_p_mel, pairs_mel = run_friedman_pivot(
    subset_mel,
    index_cols=["Audio_File", "Hop_Length", "Threshold"],
    pivot_col="Mel_Bands",
    values_col="SER",
    levels=levels_mel
)

anova_mel = df[df["Mel_Bands"].isin(levels_mel)]
anova_table_mel = run_anova(anova_mel, 'SER ~ C(Mel_Bands) + C(Audio_File)')

print("\n-- Mel_Bands (10 vs 20 vs 30 vs 40) --")
if friedman_stat_mel is not None:
    print(f"Friedman: Statistic={friedman_stat_mel:.4f}, p-value={friedman_p_mel:.4f}, pairs={pairs_mel}")
else:
    print("Friedman: Insufficient data")
print(anova_table_mel)
