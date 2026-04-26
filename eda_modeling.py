"""
BMW Used Car Price Analysis & Prediction
=========================================
End-to-end pipeline:
  1. Data loading & inspection
  2. Exploratory Data Analysis (EDA) with visualisations
  3. Feature engineering
  4. Random Forest regression for price prediction
  5. Model evaluation & feature importance
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3f55",
    "axes.labelcolor":  "#c8cce0",
    "xtick.color":      "#8b90a8",
    "ytick.color":      "#8b90a8",
    "text.color":       "#e2e5f1",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

BMW_BLUE   = "#1c6ef3"
BMW_WHITE  = "#e2e5f1"
BMW_SILVER = "#8b90a8"
ACCENT     = "#f07b3f"
PALETTE    = [BMW_BLUE, ACCENT, "#6c63ff", "#2ec4b6", "#e84393", "#f9c74f"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bmw.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & INSPECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = df["model"].str.strip()
    return df


def inspect_data(df: pd.DataFrame) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  BMW USED-CAR DATASET  ─  INSPECTION")
    print(sep)
    print(f"  Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"\n  Numeric summary:\n")
    print(df.describe().round(2).to_string())
    print(f"\n  Categorical distributions:")
    for col in ["model", "transmission", "fuelType"]:
        print(f"\n  [{col}]")
        print(df[col].value_counts().to_string())
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EDA VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    print(f"  ✔  Saved → {path}")
    plt.close(fig)


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Histograms + KDE for all numeric features."""
    cols = ["price", "mileage", "year", "tax", "mpg", "engineSize"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BMW Used Cars — Numerical Feature Distributions",
                 fontsize=16, fontweight="bold", color=BMW_WHITE, y=1.01)

    for ax, col in zip(axes.flat, cols):
        data = df[col].dropna()
        ax.hist(data, bins=40, color=BMW_BLUE, alpha=0.75, edgecolor="none", density=True)
        data.plot.kde(ax=ax, color=ACCENT, linewidth=2)
        ax.set_title(col, fontweight="bold", color=BMW_WHITE)
        ax.set_xlabel(col, color=BMW_SILVER)
        ax.set_ylabel("Density", color=BMW_SILVER)
        ax.axvline(data.median(), color="#f9c74f", linestyle="--",
                   linewidth=1.2, label=f"Median: {data.median():,.0f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "01_numerical_distributions.png")


def plot_categorical_counts(df: pd.DataFrame) -> None:
    """Count plots for model, transmission, fuelType."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("BMW Used Cars — Categorical Feature Counts",
                 fontsize=16, fontweight="bold", color=BMW_WHITE)

    # Model (sorted by count)
    model_counts = df["model"].value_counts()
    axes[0].barh(model_counts.index, model_counts.values,
                 color=BMW_BLUE, edgecolor="none")
    axes[0].set_title("Listings by Model", fontweight="bold", color=BMW_WHITE)
    axes[0].set_xlabel("Count", color=BMW_SILVER)
    axes[0].invert_yaxis()

    # Transmission
    trans_counts = df["transmission"].value_counts()
    axes[1].bar(trans_counts.index, trans_counts.values,
                color=PALETTE[:len(trans_counts)], edgecolor="none")
    axes[1].set_title("Transmission Type", fontweight="bold", color=BMW_WHITE)
    axes[1].set_ylabel("Count", color=BMW_SILVER)

    # Fuel Type
    fuel_counts = df["fuelType"].value_counts()
    wedge_props = dict(linewidth=2, edgecolor="#0f1117")
    axes[2].pie(fuel_counts.values, labels=fuel_counts.index,
                colors=PALETTE[:len(fuel_counts)],
                autopct="%1.1f%%", startangle=140,
                textprops={"color": BMW_WHITE, "fontsize": 10},
                wedgeprops=wedge_props)
    axes[2].set_title("Fuel Type Share", fontweight="bold", color=BMW_WHITE)

    plt.tight_layout()
    _save(fig, "02_categorical_counts.png")


def plot_price_by_category(df: pd.DataFrame) -> None:
    """Box plots — price split by transmission and fuel type."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Price Distribution by Category",
                 fontsize=16, fontweight="bold", color=BMW_WHITE)

    for ax, col, title in zip(
        axes,
        ["transmission", "fuelType"],
        ["Price by Transmission", "Price by Fuel Type"],
    ):
        order = df.groupby(col)["price"].median().sort_values(ascending=False).index
        sns.boxplot(
            data=df, x=col, y="price", order=order, ax=ax,
            palette=PALETTE[:df[col].nunique()],
            flierprops=dict(marker="o", markersize=2, alpha=0.3,
                            markerfacecolor=BMW_SILVER),
        )
        ax.set_title(title, fontweight="bold", color=BMW_WHITE)
        ax.set_xlabel(col, color=BMW_SILVER)
        ax.set_ylabel("Price (£)", color=BMW_SILVER)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    plt.tight_layout()
    _save(fig, "03_price_by_category.png")


def plot_scatter_relationships(df: pd.DataFrame) -> None:
    """Scatter plots: mileage vs price, year vs price, engineSize vs price."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Price Relationships — Key Numeric Features",
                 fontsize=16, fontweight="bold", color=BMW_WHITE)

    pairs = [
        ("mileage", "price", "Mileage vs Price"),
        ("year",    "price", "Year vs Price"),
        ("engineSize", "price", "Engine Size vs Price"),
    ]

    for ax, (x_col, y_col, title) in zip(axes, pairs):
        sc = ax.scatter(
            df[x_col], df[y_col],
            c=df["year"], cmap="plasma",
            alpha=0.25, s=8, linewidths=0,
        )
        ax.set_title(title, fontweight="bold", color=BMW_WHITE)
        ax.set_xlabel(x_col, color=BMW_SILVER)
        ax.set_ylabel("Price (£)", color=BMW_SILVER)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Year", color=BMW_SILVER, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=BMW_SILVER)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=BMW_SILVER)

    plt.tight_layout()
    _save(fig, "04_scatter_relationships.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Correlation Matrix — Numeric Features",
                 fontsize=15, fontweight="bold", color=BMW_WHITE)

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 10},
        linewidths=0.5, linecolor="#0f1117",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")


def plot_avg_price_by_model(df: pd.DataFrame) -> None:
    """Average price per BMW model (sorted)."""
    avg = (
        df.groupby("model")["price"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
    )
    avg.columns = ["avg_price", "count"]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle("Average Listed Price by BMW Model",
                 fontsize=15, fontweight="bold", color=BMW_WHITE)

    bars = ax.barh(avg.index, avg["avg_price"],
                   color=BMW_BLUE, edgecolor="none", height=0.65)
    # Colour top 5 with accent
    for bar in bars[-5:]:
        bar.set_color(ACCENT)

    for bar, (_, row) in zip(bars, avg.iterrows()):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"£{row['avg_price']:,.0f}  (n={int(row['count'])})",
                va="center", fontsize=8.5, color=BMW_SILVER)

    ax.set_xlabel("Average Price (£)", color=BMW_SILVER)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlim(right=avg["avg_price"].max() * 1.30)
    plt.tight_layout()
    _save(fig, "06_avg_price_by_model.png")


def plot_price_over_years(df: pd.DataFrame) -> None:
    """Median price trend over manufacturing year, split by fuel type."""
    yearly = (
        df.groupby(["year", "fuelType"])["price"]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Median Price Trend by Year & Fuel Type",
                 fontsize=15, fontweight="bold", color=BMW_WHITE)

    for i, fuel in enumerate(df["fuelType"].unique()):
        sub = yearly[yearly["fuelType"] == fuel]
        ax.plot(sub["year"], sub["price"], marker="o", markersize=5,
                linewidth=2, label=fuel, color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("Year of Manufacture", color=BMW_SILVER)
    ax.set_ylabel("Median Price (£)", color=BMW_SILVER)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(title="Fuel Type", title_fontsize=9)
    plt.tight_layout()
    _save(fig, "07_price_trend_by_year_fuel.png")


def run_eda(df: pd.DataFrame) -> None:
    print("\n[EDA] Generating visualisations …")
    plot_numerical_distributions(df)
    plot_categorical_counts(df)
    plot_price_by_category(df)
    plot_scatter_relationships(df)
    plot_correlation_heatmap(df)
    plot_avg_price_by_model(df)
    plot_price_over_years(df)
    print("[EDA] All charts saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived features
    df["car_age"]       = 2024 - df["year"]
    df["age_x_mileage"] = df["car_age"] * df["mileage"]  # interaction term
    df["log_mileage"]   = np.log1p(df["mileage"])
    df["log_price"]     = np.log1p(df["price"])

    # Encode categoricals
    le = LabelEncoder()
    for col in ["model", "transmission", "fuelType"]:
        df[f"{col}_enc"] = le.fit_transform(df[col])

    # Remove extreme outliers (mpg > 200 are data errors)
    df = df[df["mpg"] < 200].copy()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    "car_age", "log_mileage", "engineSize", "tax", "mpg",
    "model_enc", "transmission_enc", "fuelType_enc",
    "age_x_mileage",
]
TARGET = "price"


def train_evaluate(df: pd.DataFrame) -> dict:
    df_feat = engineer_features(df)
    X = df_feat[FEATURES]
    y = df_feat[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge Regression":    Ridge(alpha=10),
        "Gradient Boosting":   GradientBoostingRegressor(
                                   n_estimators=200, max_depth=5,
                                   learning_rate=0.1, random_state=42),
        "Random Forest":       RandomForestRegressor(
                                   n_estimators=300, max_depth=None,
                                   min_samples_leaf=2, n_jobs=-1,
                                   random_state=42),
    }

    results = {}
    print("[MODEL] Training …")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        results[name] = {
            "model":   model,
            "preds":   preds,
            "y_test":  y_test,
            "RMSE":    rmse,
            "MAE":     mae,
            "R2":      r2,
        }
        print(f"  {name:22s}  RMSE=£{rmse:,.0f}  MAE=£{mae:,.0f}  R²={r2:.4f}")

    return results


def plot_model_results(results: dict) -> None:
    best_name = max(results, key=lambda k: results[k]["R2"])
    best      = results[best_name]

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f"Model Evaluation — {best_name}  (R²={best['R2']:.4f})",
                 fontsize=15, fontweight="bold", color=BMW_WHITE)

    gs = gridspec.GridSpec(1, 3, figure=fig)

    # ── Actual vs Predicted ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(best["y_test"], best["preds"],
                alpha=0.2, s=7, color=BMW_BLUE, linewidths=0)
    lim = max(best["y_test"].max(), best["preds"].max())
    ax1.plot([0, lim], [0, lim], "--", color=ACCENT, linewidth=1.5)
    ax1.set_xlabel("Actual Price (£)", color=BMW_SILVER)
    ax1.set_ylabel("Predicted Price (£)", color=BMW_SILVER)
    ax1.set_title("Actual vs Predicted", fontweight="bold", color=BMW_WHITE)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    # ── Residuals ──
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = best["y_test"].values - best["preds"]
    ax2.scatter(best["preds"], residuals,
                alpha=0.2, s=7, color=ACCENT, linewidths=0)
    ax2.axhline(0, color=BMW_WHITE, linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Predicted Price (£)", color=BMW_SILVER)
    ax2.set_ylabel("Residual (£)", color=BMW_SILVER)
    ax2.set_title("Residual Plot", fontweight="bold", color=BMW_WHITE)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    # ── Feature Importances ──
    ax3 = fig.add_subplot(gs[0, 2])
    rf = best["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    colors = [ACCENT if v == importances.max() else BMW_BLUE for v in importances]
    importances.plot.barh(ax=ax3, color=colors, edgecolor="none")
    ax3.set_title("Feature Importances", fontweight="bold", color=BMW_WHITE)
    ax3.set_xlabel("Importance", color=BMW_SILVER)

    plt.tight_layout()
    _save(fig, "08_model_evaluation.png")


def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing R² across all models."""
    names  = list(results.keys())
    r2s    = [results[n]["R2"]   for n in names]
    rmses  = [results[n]["RMSE"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold", color=BMW_WHITE)

    bar_colors = [BMW_BLUE, "#6c63ff", ACCENT]
    axes[0].bar(names, r2s, color=bar_colors, edgecolor="none")
    axes[0].set_ylabel("R² Score", color=BMW_SILVER)
    axes[0].set_title("R² (higher = better)", fontweight="bold", color=BMW_WHITE)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", color=BMW_WHITE, fontsize=11)

    axes[1].bar(names, rmses, color=bar_colors, edgecolor="none")
    axes[1].set_ylabel("RMSE (£)", color=BMW_SILVER)
    axes[1].set_title("RMSE (lower = better)", fontweight="bold", color=BMW_WHITE)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 50, f"£{v:,.0f}", ha="center", color=BMW_WHITE, fontsize=10)

    plt.tight_layout()
    _save(fig, "09_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("   BMW USED-CAR ANALYSIS & PRICE PREDICTION")
    print("═" * 60)

    df = load_data(DATA_PATH)
    inspect_data(df)

    run_eda(df)

    results = train_evaluate(df)

    print("\n[PLOTS] Saving model evaluation charts …")
    plot_model_results(results)
    plot_model_comparison(results)

    best_name = max(results, key=lambda k: results[k]["R2"])
    best = results[best_name]
    print(f"""
╔══════════════════════════════════════════════════╗
║           FINAL RESULTS  ({best_name})
╠══════════════════════════════════════════════════╣
║  R²   : {best['R2']:.4f}
║  RMSE : £{best['RMSE']:,.0f}
║  MAE  : £{best['MAE']:,.0f}
╚══════════════════════════════════════════════════╝
    """)
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}\n")


if __name__ == "__main__":
    main()
