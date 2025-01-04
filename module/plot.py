import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
from scipy import stats

def plot(pred, true, save_path):

    df = pd.concat([true, pred], axis=1)

    corr = df.corr().iloc[0, 1]
    t = abs(corr) * math.sqrt(len(df) - 2) / math.sqrt(1 - corr ** 2)
    p = 2 * (1 - stats.t.cdf(t, len(df) - 2))
    plt.title(f"相関係数: {corr:.3f}")
    plt.scatter(pred, true, c="black")
    plt.xlabel("青春情報エントロピー")
    plt.ylabel("平均嗜好度")

    plt.savefig(save_path)

    print(f"相関係数: {corr:.3f}, t値: {t:.3f}, p値: {p:.3f}")