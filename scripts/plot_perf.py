import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path(__file__).with_name("results.csv")
    out_path = Path(__file__).with_name("perf.png")

    df = pd.read_csv(csv_path)

    df["size"] = df.apply(lambda r: f"{int(r.M)}x{int(r.N)}x{int(r.K)}", axis=1)
    df["work"] = df["M"] * df["N"] * df["K"]
    df = (df.sort_values(["size", "name", "GFLOPS"]).groupby(["size", "name"], as_index=False).tail(1))

    pivot = (df.pivot(index="size", columns="name", values="GFLOPS").reindex(df.sort_values("work")["size"].unique()))

    ax = pivot.plot(kind="bar", figsize=(9, 4), rot=45)
    ax.set_ylabel("GFLOPS")
    ax.set_xlabel("Problem size (M×N×K)")
    ax.set_title("CUDA MatMul Performance")
    ax.legend(title="Kernel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

if __name__ == "__main__":
    main()
