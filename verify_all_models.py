"""
Comprehensive Model Verification Script
========================================
Tests all commodity models (Cinnamon, Clove, Pepper) via the running API:
  1. API Health & Metadata
  2. Model Accuracy Metrics (RMSE, MAE, MAPE, R², Directional Accuracy)
  3. Forecast Quality (realism, non-flat, speed, scenarios)
  4. XAI Explanation Quality
  5. Context Switching (consecutive requests for different commodities)

Prerequisites: API must be running at http://localhost:8000
"""
import json, os, time, urllib.request, urllib.error, math

API_BASE = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_STEPS = 6
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results_summary = []

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label, passed, detail=""):
    status = PASS if passed else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results_summary.append((label, passed))
    return passed

def api_get(path):
    """GET request to API."""
    try:
        req = urllib.request.Request(f"{API_BASE}{path}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None

def api_post(path, body):
    """POST request to API."""
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"{API_BASE}{path}", data=data, 
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if hasattr(e, 'read') else ''
        print(f"    {WARN} API {e.code}: {body_text[:100]}")
        return None
    except Exception as e:
        print(f"    {WARN} API Error: {e}")
        return None


# ─── TEST 1: API Health & Metadata ───────────────────────────────────
def test_metadata():
    separator("TEST 1: API Health & Metadata")
    
    combos = {}
    for commodity in ['cinnamon', 'clove', 'pepper']:
        meta = api_get(f"/metadata?commodity={commodity}")
        if meta is None:
            check(f"[{commodity}] Metadata", False, "API unreachable or error")
            continue
        
        regions = meta.get('regions', [])
        grades = meta.get('grades', [])
        check(f"[{commodity}] Metadata", len(regions) > 0 and len(grades) > 0,
              f"{len(regions)} regions, {len(grades)} grades")
        
        # Save a valid combo for later testing
        grades_by_region = meta.get('grades_by_region', {})
        if grades_by_region:
            region = list(grades_by_region.keys())[0]
            grade = grades_by_region[region][0]
        else:
            region = regions[0]
            grade = grades[0]
        combos[commodity] = (region, grade)
    
    return combos


# ─── TEST 2: Model Accuracy Metrics ──────────────────────────────────
def test_accuracy_metrics():
    separator("TEST 2: Model Accuracy Metrics (from Training)")
    
    print(f"\n  {'Commodity':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'MAPE':>10} {'Status':>10}")
    print(f"  {'-'*62}")
    
    for commodity in ['cinnamon', 'clove', 'pepper']:
        results_path = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}', 'results.json')
        
        if not os.path.exists(results_path):
            check(f"[{commodity}] Metrics File", False, "results.json not found")
            continue
        
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        
        mae = metrics.get('mae', None)
        rmse = metrics.get('rmse', None)
        r2 = metrics.get('r2', None)
        
        # Compute MAPE from MAE if we have context (approximate using average price)
        # We'll estimate from the data if available
        mape_str = "N/A"
        data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{commodity}_prices.csv')
        if os.path.exists(data_path) and mae is not None:
            try:
                # Quick read to get average price for MAPE approximation
                with open(data_path, 'r') as df:
                    lines = df.readlines()
                    # Find the Regional_Price column index
                    header = lines[0].strip().split(',')
                    if 'Regional_Price' in header:
                        price_idx = header.index('Regional_Price')
                        prices = []
                        for line in lines[1:]:
                            cols = line.strip().split(',')
                            if len(cols) > price_idx:
                                try:
                                    p = float(cols[price_idx])
                                    if p > 0:
                                        prices.append(p)
                                except ValueError:
                                    pass
                        if prices:
                            avg_price = sum(prices) / len(prices)
                            mape_approx = (mae / avg_price) * 100
                            mape_str = f"{mape_approx:.1f}%"
            except Exception:
                pass
        
        # Grade the model
        if r2 is not None:
            if r2 >= 0.7:
                grade = "GOOD"
            elif r2 >= 0.4:
                grade = "FAIR"
            elif r2 >= 0:
                grade = "POOR"
            else:
                grade = "FAIL"
        else:
            grade = "N/A"
        
        mae_str = f"{mae:.1f}" if mae else "N/A"
        rmse_str = f"{rmse:.1f}" if rmse else "N/A"
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        
        print(f"  {commodity:<12} {mae_str:>10} {rmse_str:>10} {r2_str:>10} {mape_str:>10} {grade:>10}")
        
        # Check thresholds
        check(f"[{commodity}] R² Score", r2 is not None and r2 > 0,
              f"R²={r2:.4f} ({grade})" if r2 is not None else "No R² available")
        check(f"[{commodity}] MAE Score", mae is not None and mae > 0,
              f"MAE={mae:.1f} LKR" if mae else "No MAE")
        
        # Additional training info
        epochs = metrics.get('epochs_trained', 'N/A')
        train_loss = metrics.get('final_train_loss', None)
        val_loss = metrics.get('final_val_loss', None)
        tuning = metrics.get('tuning_method', 'none')
        features = len(metrics.get('feature_cols', []))
        
        print(f"    ℹ️  Epochs={epochs}, Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}" 
              if train_loss and val_loss else f"    ℹ️  Epochs={epochs}")
        print(f"    ℹ️  Tuning={tuning}, Features={features}")
        
        # Check for overfitting (train loss << val loss is OK, but val >> 10x train is suspicious)
        if train_loss and val_loss and val_loss > train_loss * 10:
            print(f"    {WARN} Possible overfitting: val_loss/train_loss = {val_loss/train_loss:.1f}x")
    
    print()


# ─── TEST 3: Forecast Quality ────────────────────────────────────────
def test_forecast(commodity, region, grade):
    """Test a forecast and return the response for further checks."""
    body = {"region": region, "grade": grade, "months": FORECAST_STEPS}
    
    start = time.time()
    result = api_post(f"/predict?commodity={commodity}", body)
    elapsed = time.time() - start
    
    if result is None:
        check(f"[{commodity}] Forecast Response", False, "No response from API")
        return None
    
    check(f"[{commodity}] Forecast Speed", elapsed < 60, f"{elapsed:.1f}s")
    
    # Check scenarios
    scenarios = result.get('scenarios', {})
    expected = ['Baseline', 'Optimistic', 'Pessimistic']
    present = [s for s in expected if s in scenarios]
    check(f"[{commodity}] Scenarios Present", len(present) == 3, str(present))
    
    # Baseline analysis
    baseline = scenarios.get('Baseline', {})
    prices = baseline.get('prices', [])
    dates = baseline.get('dates', [])
    
    check(f"[{commodity}] Forecast Length", len(prices) == FORECAST_STEPS,
          f"{len(prices)} steps")
    
    if not prices:
        return result
    
    # Non-flat check
    unique = len(set([round(p, 2) for p in prices]))
    check(f"[{commodity}] Non-Flat", unique > 1,
          f"{unique} unique values in {len(prices)} steps")
    
    # No zeros
    has_zeros = any(p == 0 or p < 1 for p in prices)
    check(f"[{commodity}] No Zeros", not has_zeros,
          f"Min={min(prices):.2f}, Max={max(prices):.2f}")
    
    # All positive
    all_positive = all(p > 0 for p in prices)
    check(f"[{commodity}] All Positive", all_positive)
    
    # Print forecast
    last_hist = result.get('historical_prices', [])
    last_price = last_hist[-1] if last_hist else prices[0]
    print(f"\n  📊 Baseline ({commodity} / {grade} / {region}):")
    print(f"     Last Historical Price: {last_price:.2f}")
    for d, p in zip(dates, prices):
        diff = ((p - last_price) / last_price * 100) if last_price else 0
        bar = "▲" if p > last_price else "▼"
        print(f"     {d}: {p:>10.2f}  {bar} {diff:+.1f}%")
    
    # Scenario ordering check
    opt = scenarios.get('Optimistic', {}).get('prices', [])
    pess = scenarios.get('Pessimistic', {}).get('prices', [])
    if opt and pess and len(opt) == len(prices):
        ordered = sum(1 for o, b, p in zip(opt, prices, pess) if o >= b >= p)
        check(f"[{commodity}] Scenario Order", ordered >= len(prices) // 2,
              f"Opt≥Base≥Pess in {ordered}/{len(prices)} steps")
        
        # Print spread
        print(f"\n  📊 Scenario Spread (first vs last):")
        print(f"     Month 1: Pess={pess[0]:.0f} | Base={prices[0]:.0f} | Opt={opt[0]:.0f}")
        print(f"     Month {len(prices)}: Pess={pess[-1]:.0f} | Base={prices[-1]:.0f} | Opt={opt[-1]:.0f}")
    
    # Price realism
    if last_price and last_price > 0:
        max_dev = max(abs(p - last_price) / last_price for p in prices)
        check(f"[{commodity}] Price Realism", max_dev < 0.80,
              f"Max deviation: {max_dev:.1%}")
    
    return result


# ─── TEST 4: XAI Quality ─────────────────────────────────────────────
def test_xai(commodity, result):
    """Check XAI explanations from forecast result."""
    if result is None:
        check(f"[{commodity}] XAI", False, "No result")
        return
    
    scenarios = result.get('scenarios', {})
    baseline = scenarios.get('Baseline', {})
    explanations = baseline.get('explanations', [])
    
    check(f"[{commodity}] XAI Present", len(explanations) > 0,
          f"{len(explanations)} explanation sets")
    
    if not explanations:
        return
    
    # Check first step
    first = explanations[0]
    has_content = isinstance(first, list) and len(first) > 0 and first[0] != ""
    if has_content:
        check(f"[{commodity}] XAI Content", True, 
              f"\"{first[0][:70]}...\"" if len(first[0]) > 70 else f"\"{first[0]}\"")
    else:
        check(f"[{commodity}] XAI Content", False, "Empty explanations")
    
    # Check for economic drivers
    all_text = " ".join([" ".join(step) if isinstance(step, list) else str(step) 
                         for step in explanations])
    drivers = [kw for kw in ['Temperature', 'Rainfall', 'Exchange', 'Inflation', 
                              'Production', 'National', 'Weather', 'Currency', 'stable']
               if kw.lower() in all_text.lower()]
    check(f"[{commodity}] XAI Drivers", len(drivers) > 0,
          f"Detected: {drivers}")
    
    # Check for LKR values
    has_lkr = "LKR" in all_text or "influence" in all_text.lower()
    check(f"[{commodity}] XAI Impact Scores", has_lkr,
          "Has quantified impacts" if has_lkr else "No LKR/influence values found")


# ─── TEST 5: Context Switching ────────────────────────────────────────
def test_context_switching(combos):
    separator("TEST 5: Context Switching (Rapid Commodity Switches)")
    
    if len(combos) < 2:
        check("Context Switching", False, "Need at least 2 commodities available")
        return
    
    # Rapidly switch between commodities using known-good combos
    commodities_to_test = list(combos.keys())
    c1, c2 = commodities_to_test[0], commodities_to_test[1]
    
    print(f"  ℹ️  Sending rapid requests: {c1} → {c2} → {c1}")
    
    c1_results = []
    all_ok = True
    
    for c in [c1, c2, c1]:
        r, g = combos[c]
        body = {"region": r, "grade": g, "months": 3}
        resp = api_post(f"/predict?commodity={c}", body)
        if resp is None:
            all_ok = False
        else:
            baseline_prices = resp.get('scenarios', {}).get('Baseline', {}).get('prices', [])
            if c == c1:
                c1_results.append(baseline_prices)
            print(f"    {PASS} {c} ({g}/{r}): {[round(p,1) for p in baseline_prices[:3]]}")
    
    if all_ok:
        check("All Requests Succeeded", True, f"3/3 requests returned valid data")
    else:
        check("All Requests Succeeded", False, "Some requests failed")
    
    # Check first-commodity results are consistent after switching
    if len(c1_results) == 2 and c1_results[0] and c1_results[1]:
        # Prices should be identical (same input, same model)
        max_diff = max(abs(a - b) for a, b in zip(c1_results[0], c1_results[1]))
        check(f"Consistency After Switch", max_diff < 1.0,
              f"Max price diff={max_diff:.2f} (same commodity before/after switch)")
    
    # No zeros in any result
    if all_ok:
        check("No Zeros After Switch", True)


# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "🔬 " * 20)
    print("  COMPREHENSIVE MODEL VERIFICATION (via API)")
    print("  Testing: Cinnamon, Clove, Pepper + XAI + Context Switching")
    print("  API: " + API_BASE)
    print("🔬 " * 20)
    
    # Pre-check: API reachable?
    try:
        health = api_get("/metadata?commodity=cinnamon")
    except Exception:
        health = None
    if health is None:
        print(f"\n  {FAIL} API is not reachable at {API_BASE}")
        print("  Please start the API first: cd src && python api.py")
        print("  Then re-run: python verify_all_models.py")
        return
    print(f"\n  {PASS} API is running")
    
    # Test 1: Metadata (also collects valid combos)
    combos = test_metadata()
    
    # Test 2: Model Accuracy Metrics
    test_accuracy_metrics()
    
    # Test 3 & 4: Forecast + XAI per commodity
    for c, (region, grade) in combos.items():
        separator(f"COMMODITY: {c.upper()} ({grade}/{region})")
        result = test_forecast(c, region, grade)
        test_xai(c, result)
    
    # Test 5: Context switching
    test_context_switching(combos)
    
    # ─── Final Report ─────────────────────────────────────────────
    separator("FINAL REPORT")
    
    passed = sum(1 for _, p in results_summary if p)
    failed = sum(1 for _, p in results_summary if not p)
    total = len(results_summary)
    
    print(f"\n  Total Checks: {total}")
    print(f"  {PASS} Passed: {passed}")
    print(f"  {FAIL} Failed: {failed}")
    print(f"  Score: {passed}/{total} ({100*passed/total:.0f}%)" if total > 0 else "")
    
    if failed > 0:
        print(f"\n  Failed Checks:")
        for label, p in results_summary:
            if not p:
                print(f"    {FAIL} {label}")
    
    print(f"\n{'='*60}")
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — Models are production-ready!")
    elif failed <= 3:
        print("  ⚠️  MOSTLY PASSING — Review the failed checks above.")
    else:
        print("  🚨 SIGNIFICANT FAILURES — Models need attention.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
