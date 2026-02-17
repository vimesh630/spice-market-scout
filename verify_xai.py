"""
Verification script for XAI Explainable AI upgrade.
Loads real commodity models and validates explanation output format.
"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src import forecasting_engine as engine
import pandas as pd

def verify_xai(commodity='cinnamon', steps=3):
    print(f"\n{'='*60}")
    print(f"  XAI Verification — {commodity.upper()}")
    print(f"{'='*60}")

    # 1. Load model artifacts
    model = engine.load_artifacts(commodity)
    if model is None:
        print(f"❌ Could not load model for {commodity}. Skipping.")
        return False

    # 2. Load and preprocess data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'processed', f'{commodity}_prices.csv')
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False

    raw_df = pd.read_csv(data_path)
    
    # Smart filtering: pick the grade/region combo with the MOST rows
    best_df = raw_df
    chosen_grade = "N/A"
    chosen_region = "N/A"
    
    if 'Grade' in raw_df.columns and 'Region' in raw_df.columns:
        combos = raw_df.groupby(['Grade', 'Region']).size().reset_index(name='count')
        combos = combos.sort_values('count', ascending=False)
        if len(combos) > 0:
            chosen_grade = combos.iloc[0]['Grade']
            chosen_region = combos.iloc[0]['Region']
            best_df = raw_df[(raw_df['Grade'] == chosen_grade) & (raw_df['Region'] == chosen_region)]
    elif 'Grade' in raw_df.columns:
        counts = raw_df['Grade'].value_counts()
        chosen_grade = counts.index[0]
        best_df = raw_df[raw_df['Grade'] == chosen_grade]
    elif 'Region' in raw_df.columns:
        counts = raw_df['Region'].value_counts()
        chosen_region = counts.index[0]
        best_df = raw_df[raw_df['Region'] == chosen_region]

    df = engine.preprocess_data(best_df.copy(), training_mode=False)
    if 'Date' in df.columns:
        df = df.sort_values('Date')

    if len(df) < engine.SEQUENCE_LENGTH:
        print(f"❌ Not enough rows ({len(df)}) for {commodity} [{chosen_grade}/{chosen_region}]. Need {engine.SEQUENCE_LENGTH}.")
        return False

    print(f"✅ Loaded {len(df)} rows for {commodity} (Grade: {chosen_grade}, Region: {chosen_region})")

    # 3. Run forecast
    scenarios = engine.forecast_multistep(model, df, steps=steps, commodity=commodity)

    # 4. Validate explanations
    baseline = scenarios.get('Baseline', {})
    dates = baseline.get('dates', [])
    prices = baseline.get('prices', [])
    explanations = baseline.get('explanations', [])

    print(f"\n--- Baseline Forecast ({steps} months) ---")
    all_pass = True
    for i in range(len(dates)):
        date_str = dates[i]
        price = prices[i]
        month_expls = explanations[i] if i < len(explanations) else []
        print(f"\n  📅 {date_str}  →  {price:.2f} LKR")
        if not isinstance(month_expls, list) or len(month_expls) == 0:
            print(f"    ❌ FAIL: Explanations missing or not a list!")
            all_pass = False
            continue
        for expl in month_expls:
            if not isinstance(expl, str):
                print(f"    ❌ FAIL: Explanation is not a string: {expl}")
                all_pass = False
            else:
                # Stable market is a valid fallback
                if "stable market" in expl.lower():
                    print(f"    ✅ {expl}")
                else:
                    # Check for new format markers
                    has_influence = "influence" in expl.lower()
                    has_lkr = "LKR" in expl
                    has_driven = "driven" in expl.lower()
                    fmt_ok = has_influence and has_lkr and has_driven
                    marker = "✅" if fmt_ok else "⚠️"
                    print(f"    {marker} {expl}")
                    if not fmt_ok:
                        all_pass = False

    # Verify all 3 scenarios have explanations
    print(f"\n--- Scenario Summary ---")
    for name in ['Baseline', 'Optimistic', 'Pessimistic']:
        sc = scenarios.get(name, {})
        n_expls = len(sc.get('explanations', []))
        n_prices = len(sc.get('prices', []))
        status = "✅" if n_expls == n_prices else "❌"
        print(f"  {status} {name}: {n_prices} prices, {n_expls} explanation sets")
        if n_expls != n_prices:
            all_pass = False

    if all_pass:
        print(f"\n🎉 All XAI checks PASSED for {commodity}!")
    else:
        print(f"\n⚠️ Some XAI checks had warnings for {commodity}.")
    return all_pass


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    commodities = []
    for c in ['cinnamon', 'clove', 'pepper']:
        model_path = os.path.join(base_dir, 'models', f'lstm_{c}', 'lstm_model.keras')
        if os.path.exists(model_path):
            commodities.append(c)
        else:
            print(f"⏭️ Skipping {c} — no trained model found.")

    results = {}
    for com in commodities:
        results[com] = verify_xai(com, steps=3)

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for com, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {com}")
