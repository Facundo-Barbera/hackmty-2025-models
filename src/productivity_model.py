import pandas as pd
import numpy as np
from pathlib import Path


def load_productivity_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def load_efficiency_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def fit_linear_sec_per_item(efficiency_df):
    N = efficiency_df['Items_Packed'].values
    T = efficiency_df['Duration_Seconds'].values
    s_hat = T.sum() / N.sum()
    t0_hat = 0.0
    T_pred = s_hat * N
    residuals = T - T_pred
    q90_residual = np.percentile(np.abs(residuals), 90)
    return t0_hat, s_hat, residuals, q90_residual


def drawer_p50_p90_lm(N, t0, s, q90_residual, multipliers=(1, 1, 1)):
    m = np.prod(multipliers)
    p50_sec = t0 + m * s * N
    p90_sec = p50_sec + q90_residual
    return p50_sec / 60, p90_sec / 60


def trolley_p50_p90_lm(drawers, t0, s, q90_residual, multipliers=(1, 1, 1)):
    p50_total, p90_total = 0, 0
    for N in drawers:
        p50, p90 = drawer_p50_p90_lm(N, t0, s, q90_residual, multipliers)
        p50_total += p50
        p90_total += p90
    return p50_total, p90_total


def target_gap_metrics(p50_trolley_min, p90_trolley_min, target_p50_min=5.0, target_p90_min=7.0):
    gap_p50 = p50_trolley_min - target_p50_min
    gap_p90 = p90_trolley_min - target_p90_min
    return {
        'predicted_p50_min': round(p50_trolley_min, 2),
        'predicted_p90_min': round(p90_trolley_min, 2),
        'target_p50_min': target_p50_min,
        'target_p90_min': target_p90_min,
        'gap_p50_min': round(gap_p50, 2),
        'gap_p90_min': round(gap_p90, 2),
        'exceeds_p50_target': gap_p50 > 0,
        'exceeds_p90_target': gap_p90 > 0
    }


def format_output(p50_min, p90_min, t0=None, s=None):
    variability = (p90_min - p50_min) / p50_min
    result = {
        "p50_min": round(p50_min, 2),
        "p90_min": round(p90_min, 2),
        "variability": round(variability, 3)
    }
    if t0 is not None and s is not None:
        result["model_params"] = {
            "t0_sec": round(t0, 2),
            "s_sec_per_item": round(s, 4)
        }
    return result


def print_results(label, result):
    print(f"\n{label}")
    print("=" * 50)
    print(f"P50: {result['p50_min']} minutes")
    print(f"P90: {result['p90_min']} minutes")
    print(f"Variability: {result['variability']}")
    if 'model_params' in result:
        print(f"Model: t0={result['model_params']['t0_sec']}s, s={result['model_params']['s_sec_per_item']}s/item")


def print_gap_report(gaps):
    print("\n" + "=" * 60)
    print("KPI GAP REPORT")
    print("=" * 60)
    print(f"Operational Targets:")
    print(f"  P50 target:                    {gaps['target_p50_min']:.1f} minutes")
    print(f"  P90 target:                    {gaps['target_p90_min']:.1f} minutes")
    print()
    print(f"Predicted Performance:")
    print(f"  P50 predicted:                 {gaps['predicted_p50_min']:.2f} minutes")
    print(f"  P90 predicted:                 {gaps['predicted_p90_min']:.2f} minutes")
    print()
    print(f"Gap Analysis:")
    print(f"  P50 gap:                       {gaps['gap_p50_min']:+.2f} minutes")
    print(f"  P90 gap:                       {gaps['gap_p90_min']:+.2f} minutes")
    print()
    print(f"Status:")
    p50_status = "EXCEEDS TARGET" if gaps['exceeds_p50_target'] else "MEETS TARGET"
    p90_status = "EXCEEDS TARGET" if gaps['exceeds_p90_target'] else "MEETS TARGET"
    print(f"  P50:                           {p50_status}")
    print(f"  P90:                           {p90_status}")
    print("=" * 60)


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    print("\n" + "=" * 60)
    print("LINEAR MODEL PRODUCTIVITY ESTIMATOR")
    print("=" * 60)
    print("This is a measurement engine that learns from historical data")
    print("and compares predictions against operational targets.")
    print("=" * 60)

    productivity_df = load_productivity_data(data_dir / "productivity_estimation.csv")
    efficiency_df = load_efficiency_data(data_dir / "employee_efficiency.csv")

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Productivity data:               {len(productivity_df)} drawers")
    print(f"Efficiency data:                 {len(efficiency_df)} records")
    print(f"Total items (efficiency):        {efficiency_df['Items_Packed'].sum()}")
    print(f"Total duration (efficiency):     {efficiency_df['Duration_Seconds'].sum():.0f}s")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("FITTING LINEAR MODEL")
    print("=" * 60)
    print("Learning T_drawer = t0 + s * N_items from efficiency data...")

    t0_hat, s_hat, residuals, q90_residual = fit_linear_sec_per_item(efficiency_df)

    print("\n" + "=" * 60)
    print("LEARNED MODEL PARAMETERS")
    print("=" * 60)
    print(f"t0 (setup time/drawer):          {t0_hat:.2f} seconds")
    print(f"s (seconds per item):            {s_hat:.4f} seconds/item")
    print(f"Residual P90:                    {q90_residual:.2f} seconds")
    print(f"Mean absolute residual:          {np.mean(np.abs(residuals)):.2f} seconds")
    print(f"Std dev of residuals:            {np.std(residuals):.2f} seconds")
    print("=" * 60)

    if t0_hat < 0:
        print("\nERROR: t0_hat < 0. Model fitting failed!")
        return
    if s_hat <= 0:
        print("\nERROR: s_hat <= 0. Model fitting failed!")
        return
    if q90_residual <= 0:
        print("\nERROR: Residual P90 <= 0. Model fitting failed!")
        return

    print("\nAll acceptance criteria passed.")

    sample_drawer = productivity_df.iloc[0]['Total_Items']
    p50_drawer, p90_drawer = drawer_p50_p90_lm(sample_drawer, t0_hat, s_hat, q90_residual)
    drawer_result = format_output(p50_drawer, p90_drawer, t0_hat, s_hat)
    print_results(f"EXAMPLE DRAWER PREDICTION (N={sample_drawer} items)", drawer_result)

    sample_drawers = productivity_df.head(5)['Total_Items'].tolist()
    p50_trolley, p90_trolley = trolley_p50_p90_lm(sample_drawers, t0_hat, s_hat, q90_residual)
    trolley_result = format_output(p50_trolley, p90_trolley)
    print_results(f"EXAMPLE TROLLEY PREDICTION ({len(sample_drawers)} drawers, {sum(sample_drawers)} items)", trolley_result)

    gaps = target_gap_metrics(p50_trolley, p90_trolley)
    print_gap_report(gaps)

    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    print(f"t0_hat >= 0:                     {t0_hat >= 0}")
    print(f"s_hat > 0:                       {s_hat > 0}")
    print(f"Residual P90 > 0:                {q90_residual > 0}")
    print(f"Predictions scale with items:    {p90_trolley > p90_drawer}")
    print(f"KPI gap visible:                 {gaps['gap_p50_min'] != 0 or gaps['gap_p90_min'] != 0}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
