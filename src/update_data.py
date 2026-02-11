"""
CLI tool for updating the spice market dataset.

Usage:
    # Update for a specific month
    python update_data.py --month 2025-09
    
    # Update a date range
    python update_data.py --start 2025-01 --end 2025-09
    
    # Dry run (preview without saving)
    python update_data.py --month 2025-09 --dry-run
    
    # Retrain model after update
    python update_data.py --month 2025-09 --retrain
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from data_pipeline import DataPipeline, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Update spice market dataset with latest data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python update_data.py --month 2025-09              # Update single month
    python update_data.py --start 2025-01 --end 2025-09  # Update range
    python update_data.py --month 2025-09 --dry-run    # Preview only
    python update_data.py --month 2025-09 --retrain    # Update and retrain
        """
    )
    
    parser.add_argument(
        '--month', '-m',
        type=str,
        help='Month to update (YYYY-MM format)'
    )
    parser.add_argument(
        '--start', '-s',
        type=str,
        help='Start month for range update (YYYY-MM format)'
    )
    parser.add_argument(
        '--end', '-e',
        type=str,
        help='End month for range update (YYYY-MM format)'
    )
    parser.add_argument(
        '--commodity', '-c',
        type=str,
        default='cinnamon',
        help='Commodity to update (default: cinnamon)'
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help="Preview data without saving to disk"
    )
    parser.add_argument(
        '--retrain', '-r',
        action='store_true',
        help='Retrain model after updating data'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.month and not args.start:
        # Default to current month
        today = date.today()
        args.month = today.strftime('%Y-%m')
        print(f"No date specified, defaulting to current month: {args.month}")
    
    # Determine date range
    if args.month:
        start_date = args.month
        end_date = args.month
    else:
        start_date = args.start
        end_date = args.end or args.start
    
    print(f"\n=== Spice Market Data Update ===")
    print(f"Commodity: {args.commodity}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Fail-fast check: Ensure target file is writable
    from config import PROCESSED_DATA_DIR
    target_file = os.path.join(PROCESSED_DATA_DIR, f'{args.commodity}_prices.csv')
    if os.path.exists(target_file):
        try:
            with open(target_file, 'a'):
                pass
        except PermissionError:
            print(f"ERROR: Permission denied for {target_file}.")
            print("Please close any applications using this file (e.g., Excel, api.py) and try again.")
            return 1

    # Run the pipeline
    try:
        df = run_pipeline(
            commodity=args.commodity,
            start_date=start_date,
            end_date=end_date,
            dry_run=args.dry_run
        )
        
        if df.empty:
            print("No data collected!")
            return 1
        
        print(f"\n=== Collection Complete ===")
        print(f"Rows collected: {len(df)}")
        print(f"Grades: {df['Grade'].nunique()}")
        print(f"Regions: {df['Region'].nunique()}")
        
        if args.verbose:
            print("\nSample data:")
            print(df.head(12).to_string())
        
        if args.dry_run:
            print("\n[DRY RUN] Data was not saved to disk.")
        else:
            print(f"\nData saved successfully!")
        
        # Retrain model if requested
        if args.retrain and not args.dry_run:
            print("\n=== Retraining Model ===")
            try:
                # Import and run the forecasting engine
                from forecasting_engine import train_all_models
                train_all_models(args.commodity)
                print("Model retraining complete!")
            except ImportError:
                print("Warning: Could not import forecasting_engine for retraining")
            except Exception as e:
                print(f"Error during retraining: {e}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
