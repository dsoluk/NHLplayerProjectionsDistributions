import argparse
from typing import Dict, List, Optional
# TODO merge PP, remove PK, change SIT to all
import pandas as pd

# Thin CLI imports from the package modules
from nhlproj.config.schema import NST_SCHEMA, split_columns_by_scorecat
from nhlproj.sources.nst_source import (
    build_nst_url,
    fetch_nst_playerteams,
    normalize_nst_columns,
    VALID_SITS,
)
from nhlproj.sources.excel_source import safe_read_excel
from nhlproj.analysis.fitting import analyze_dataframe
from nhlproj.scoring.scoring import compute_category_scores, scale_scores, apply_pos_group_weighted_overall
from nhlproj.utils.columns import coerce_specific_columns, sanitize_ipp, derive_pos_group

# Defaults for legacy Excel path
EXCEL_PATH = r"C:\Users\soluk\OneDrive\Documents\FantasyNHL\NatePts.xlsx"
PREFERRED_SHEET_NAME = "NatePts"

# Legacy fallbacks for Excel/Budget scenario
POINTS_CATEGORIES = ["G", "A", "PPP", "SOG"]
BANGER_CATEGORIES = ["HIT", "BLK", "PIM"]
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHL Player Projections & Distributions (thin CLI)")
    parser.add_argument("--source", choices=["excel", "nst"], default="excel", help="Data source to use")

    # Excel options
    parser.add_argument("--excel-path", default=EXCEL_PATH, help="Path to Excel file (when source=excel)")
    parser.add_argument("--sheet", default=PREFERRED_SHEET_NAME, help="Sheet name for Excel source")

    # NST options
    parser.add_argument("--fromseason", default="20252026", help="NST fromseason (YYYYYYYY)")
    parser.add_argument("--thruseason", default="20252026", help="NST thruseason (YYYYYYYY)")
    parser.add_argument("--sit", choices=VALID_SITS, default="5v5", help="NST situation: 5v5, pp, pk, all")
    parser.add_argument("--team", default="ALL", help="NST team code or ALL")
    parser.add_argument("--pos", default="S", help="NST position filter (S=skaters)")

    # Analysis/scoring options
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting of fitted distributions")
    parser.add_argument("--method", choices=["cdf_z", "mean_std"], default="cdf_z", help="Scoring method for per-value z-scores")
    parser.add_argument(
        "--scale",
        choices=["percentile", "minmax", "rank", "none"],
        default="percentile",
        help="Optional 0–100 scaling of all *_score, PointsScore, BangerScore, OverallScore for presentation",
    )
    parser.add_argument(
        "--pos-group-scope",
        choices=["ALL", "F", "D"],
        default="ALL",
        help="Limit fitting/scoring to rows in this position group. F includes B (both), D includes B.",
    )

    # Output / utilities
    parser.add_argument("--output-csv", default="", help="Optional path to save scored output CSV")
    parser.add_argument("--list-cols", action="store_true", help="List categorized columns after load and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load source
    if args.source == "excel":
        try:
            df = safe_read_excel(args.excel_path, preferred_sheet=args.sheet)
        except FileNotFoundError:
            print(f"Excel file not found at: {args.excel_path}")
            return
        except Exception as e:
            print(str(e))
            return
        # Ensure common count columns are numeric if present and sanitize IPP
        df = coerce_specific_columns(df, ["HIT", "BLK", "PIM"])
        df = sanitize_ipp(df, column="IPP")
        # Derive pos_group if Position available
        df = derive_pos_group(df, position_col="Position", out_col="pos_group")
    else:
        url = build_nst_url(
            fromseason=args.fromseason,
            thruseason=args.thruseason,
            sit=args.sit,
            team=args.team,
            pos=args.pos,
        )
        print(f"Fetching NST data from: {url}")
        try:
            raw_df = fetch_nst_playerteams(url)
        except Exception as e:
            print(str(e))
            return
        print("NST data loaded. Raw columns:")
        print(list(raw_df.columns))
        df = normalize_nst_columns(raw_df, sit=args.sit)

    print("Data loaded. Columns available:")
    print(list(df.columns))
    print()

    # Ensure TOI/GP is numeric for filtering
    if "TOI/GP" in df.columns:
        df["TOI/GP"] = pd.to_numeric(df["TOI/GP"], errors="coerce")

    # 1b) Apply position-group scope filtering for fitting/scoring
    scope = getattr(args, "pos_group_scoop", None)  # defensive default
    scope = args.pos_group_scope if hasattr(args, "pos_group_scope") else (scope or "ALL")
    if scope not in ("ALL", "F", "D"):
        scope = "ALL"
    if scope != "ALL":
        if "pos_group" not in df.columns:
            print(f"Warning: pos_group not available; cannot apply scope={scope}. Proceeding without filtering.")
            df_fit = df
        else:
            if scope == "F":
                mask_scope = df["pos_group"].isin(["F", "B"]).fillna(False)
            else:  # scope == "D"
                mask_scope = df["pos_group"].isin(["D", "B"]).fillna(False)
            kept = int(mask_scope.sum())
            print(f"Applying position-group scope={scope}: keeping {kept} of {len(df)} rows.")
            df_fit = df.loc[mask_scope].copy()
    else:
        df_fit = df
        if "pos_group" in df.columns:
            counts = df["pos_group"].value_counts(dropna=False).to_dict()
            print(f"pos_group counts: {counts}")

    # 1c) Apply TOI/GP usage thresholds based on sit and pos_group
    before_n = len(df_fit)
    if "TOI/GP" in df_fit.columns:
        sit_lower = (args.sit or "").lower() if hasattr(args, "sit") else ""
        mask_usage = pd.Series([True] * len(df_fit), index=df_fit.index)
        if sit_lower == "5v5" or sit_lower == "all":
            # Forwards/Both: >=12; Defense/Both: >=15; Both passes if either threshold is met
            if "pos_group" in df_fit.columns:
                pg = df_fit["pos_group"].fillna("")
                toi_gp = df_fit["TOI/GP"]
                m_f = pg.isin(["F", "B"]) & (toi_gp >= 12)
                m_d = pg.isin(["D", "B"]) & (toi_gp >= 15)
                mask_usage = (m_f | m_d).fillna(False)
            else:
                # No pos_group info: use the lower threshold 12 as a conservative filter
                mask_usage = (df_fit["TOI/GP"] >= 12).fillna(False)
        elif sit_lower == "pp":
            mask_usage = (df_fit["TOI/GP"] >= 3).fillna(False)
        else:
            # pk or others: no usage filter for now
            mask_usage = pd.Series([True] * len(df_fit), index=df_fit.index)

        df_fit_before_filter = df_fit.copy()
        df_fit = df_fit.loc[mask_usage].copy()
        after_n = len(df_fit)
        print(f"Applied TOI/GP usage filter for sit='{sit_lower}': kept {after_n} of {before_n} rows.")
        if "pos_group" in df_fit.columns:
            counts_after = df_fit["pos_group"].value_counts(dropna=False).to_dict()
            print(f"pos_group counts after usage filter: {counts_after}")
        # For 5v5, print positive fractions for key stats pre/post
        if sit_lower == "5v5" and set(["Goals/60", "Total Assists/60"]).issubset(df_fit_before_filter.columns):
            def pos_frac(frame, col):
                s = pd.to_numeric(frame[col], errors="coerce")
                return float((s > 0).mean()) if len(frame) else float('nan')
            g_pre = pos_frac(df_fit_before_filter, "Goals/60")
            a_pre = pos_frac(df_fit_before_filter, "Total Assists/60")
            g_post = pos_frac(df_fit, "Goals/60")
            a_post = pos_frac(df_fit, "Total Assists/60")
            print(f"Positive fractions Goals/60 pre/post: {g_pre:.3f} -> {g_post:.3f}")
            print(f"Positive fractions Total Assists/60 pre/post: {a_pre:.3f} -> {a_post:.3f}")
    else:
        print("Warning: 'TOI/GP' not found; usage filter not applied.")

    # 2) Determine schema-controlled columns (NST) or legacy fallbacks (Excel)
    if args.source == "nst" and NST_SCHEMA:
        cols = list(df_fit.columns)
        points_cols, banger_cols, analyze_cols = split_columns_by_scorecat(cols, NST_SCHEMA)
        info_only_present = [c for c in cols if NST_SCHEMA.get(c) == "InfoOnly"]
        other_present = [c for c in cols if NST_SCHEMA.get(c) == "Other"]
        print("Schema categorization (present columns):")
        print(f"  Points: {points_cols}")
        print(f"  Banger: {banger_cols}")
        print(f"  Other: {other_present}")
        print(f"  InfoOnly: {info_only_present}")
        # PP-specific override: analyze only Total Points/60 on power play
        sit_lower2 = (args.sit or "").lower() if hasattr(args, "sit") else ""
        if sit_lower2 == "pp":
            if "Total Points/60" in df_fit.columns:
                points_cols = ["Total Points/60"]
                banger_cols = []
                analyze_cols = ["Total Points/60"]
                print("PP mode: restricting analysis to ['Total Points/60'].")
            else:
                print("PP mode: 'Total Points/60' not found; no analysis will be performed.")
                analyze_cols = []
        if args.list_cols:
            return
        if analyze_cols is not None and len(analyze_cols) == 0:
            print("No schema-mapped columns found to analyze in NST data after PP override.")
        if not analyze_cols:
            print("No schema-mapped columns found to analyze in NST data. Proceeding with numeric columns fallback.")
            analyze_cols = None
    else:
        points_cols, banger_cols = POINTS_CATEGORIES, BANGER_CATEGORIES
        analyze_cols = None

    # 3) Analyze numeric distributions and fit models
    best_fits = analyze_dataframe(df_fit, show_plots=not args.no_plots, only_columns=analyze_cols)

    # 4) Score and aggregate
    category_weights = DEFAULT_CATEGORY_WEIGHTS
    if best_fits:
        print("Computing standardized scores using best-by-AIC distributions...")
        scored_df = compute_category_scores(
            df_fit,
            best_fits,
            points_cols,
            banger_cols,
            weights=category_weights,
            method=args.method,
        )

        # 5) Optional presentation scaling to 0–100
        if args.scale and args.scale != "none":
            scored_df = scale_scores(scored_df, mode=args.scale)

        # 5b) Apply position-group weighted BalancedScore and set OverallScore accordingly
        weights_map = {
            'F': (0.6, 0.4),
            'D': (0.4, 0.6),
            'B': (0.5, 0.5),
            'DEFAULT': (0.5, 0.5),
        }
        scored_df = apply_pos_group_weighted_overall(scored_df, weights_map=weights_map)

        # Preview
        relevant_cols = [c for c in points_cols + banger_cols if c in df.columns]
        score_columns = [f"{c}_score" for c in relevant_cols if f"{c}_score" in scored_df.columns] + [
            "PointsScore",
            "BangerScore",
            "BalancedScore",
            "OverallScore",
        ]
        existing_score_columns = [c for c in score_columns if c in scored_df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\nScore preview (first 10 rows):")
            print(scored_df[existing_score_columns].head(10))

        # 6) Export
        output_path: Optional[str] = None
        if args.output_csv:
            output_path = args.output_csv
        elif args.source == "excel":
            output_path = r"C:\Users\soluk\Downloads\NateProj2526_scores.csv"
        if output_path:
            try:
                scored_df = sanitize_ipp(scored_df, column="IPP")
                scored_df.to_csv(output_path, index=False)
                print(f"\nScored output saved to: {output_path}")
            except Exception as e:
                print(f"\nFailed to save scored output to CSV: {e}")
    else:
        print("No fits were produced; skipping scoring.")


if __name__ == "__main__":
    main()
