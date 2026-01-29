from __future__ import annotations

import pandas as pd

from portioning.engines.base import Engine, EngineInput, EngineResult
from portioning.engines import a3d3_core


class TwoStageEngine(Engine):
    name = "two_stage"

    def run(self, inp: EngineInput) -> EngineResult:
        # The A3D3 solver expects to read Excel sheets with its own column conventions
        # (Item Number, Channel, gram ranges, required pounds, etc.).
        # Here we assume inp.df is already the sheet you want solved OR a Month_Config sheet.
        df = inp.df.copy()

        # A3D3 uses argparse args; we emulate a minimal args object.
        class Args:
            time_limit = int(inp.time_limit_sec)
            gap = float(inp.gap)
            weeks_month = float(getattr(inp, "weeks_month", 4.0) or 4.0)
            pieces_per_min = float(inp.pieces_per_min)
            line_eff = float(inp.line_eff)
            chunk_size = int(inp.chunk_size)
            write_library = False

        args = Args()

        warnings = []
        meta = {
            "time_limit_sec": args.time_limit,
            "gap": args.gap,
            "chunk_size": args.chunk_size,
            "pieces_per_min": args.pieces_per_min,
            "line_eff": args.line_eff,
        }

        # If Month_Config mode, df should be Month_Config sheet
        if inp.use_month_config:
            # A3D3 expects load_month_config(df_mc, default_weeks=...)
            if not hasattr(a3d3_core, "load_month_config") or not hasattr(a3d3_core, "solve_month_decomposed"):
                raise RuntimeError("a3d3_core module missing expected functions. Check A3D3.txt integrity.")

            month_rows, month_weeks = a3d3_core.load_month_config(df, default_weeks=args.weeks_month)

            all_exec = []
            for month, items in month_rows:
                wk = month_weeks.get(month, args.weeks_month)
                time_summary, sku_pounds, pieces_report, pattern_exec, remaining_df = a3d3_core.solve_month_decomposed(
                    items, wk, args, args.pieces_per_min, args.line_eff, chunk_size=args.chunk_size
                )
                if not pattern_exec.empty:
                    pattern_exec.insert(0, "Month", month)
                    all_exec.append(pattern_exec)

            exe_df = pd.concat(all_exec, ignore_index=True) if all_exec else pd.DataFrame()

        else:
            # Single sheet mode: df is the "items" sheet.
            if not hasattr(a3d3_core, "load_items_sheet") or not hasattr(a3d3_core, "solve_month_decomposed"):
                # Older variant: A3D3.txt may define load_single_month instead; be permissive.
                raise RuntimeError("a3d3_core module missing expected functions. Check A3D3.txt integrity.")

            items, weeks_found, ppm_found, eff_found = a3d3_core.load_items_sheet(df, default_weeks=args.weeks_month)
            wk = weeks_found if weeks_found else args.weeks_month
            ppm = ppm_found if ppm_found else args.pieces_per_min
            eff = eff_found if eff_found else args.line_eff

            meta.update({"weeks_month": wk, "pieces_per_min_used": ppm, "line_eff_used": eff})

            time_summary, sku_pounds, pieces_report, exe_df, remaining_df = a3d3_core.solve_month_decomposed(
                items, wk, args, ppm, eff, chunk_size=args.chunk_size
            )

        # Convert Pattern_Execution into a "ranked results"-like table for Streamlit:
        # We treat each executed pattern as a "combination".
        if exe_df is None or exe_df.empty:
            warnings.append("Two-stage solver returned no executed patterns.")
            results = pd.DataFrame()
        else:
            # Heuristic mapping into Upgrade/Trim:
            # - A3D3 reports Total_Trim_Grams and Pounds_per_Run, Total_Randoms, etc.
            # We'll compute Trim_% as trim grams / (fillet grams) * 100, and Upgrade_% as 100 - Trim_%.
            # This is not identical to the enumeration bucket math, but gives a consistent ranking axis.
            tmp = exe_df.copy()
            if "Trim_per_Fillet" in tmp.columns:
                # Trim_per_Fillet is grams of trim per fillet (already)
                tmp["Trim_%"] = (tmp["Trim_per_Fillet"] / float(getattr(a3d3_core, "FILLET_GRAMS", 479.0))) * 100.0
                tmp["Upgrade_%"] = 100.0 - tmp["Trim_%"]
            elif "Total_Trim_Grams" in tmp.columns and "Qty" in tmp.columns:
                fillet_g = float(getattr(a3d3_core, "FILLET_GRAMS", 479.0))
                tmp["Trim_%"] = (tmp["Total_Trim_Grams"] / (tmp["Qty"].clip(lower=1) * fillet_g)) * 100.0
                tmp["Upgrade_%"] = 100.0 - tmp["Trim_%"]
            else:
                tmp["Trim_%"] = None
                tmp["Upgrade_%"] = None

            # Derive SKU_IDs / CustomerTypes if present. (A3D3 uses Item/Channel)
            if "Pattern" in tmp.columns:
                tmp["SKU_IDs"] = tmp["Pattern"].astype(str)
            elif "Items" in tmp.columns:
                tmp["SKU_IDs"] = tmp["Items"].astype(str)
            else:
                tmp["SKU_IDs"] = tmp.index.astype(str)

            if "Channel" in tmp.columns:
                tmp["CustomerTypes"] = tmp["Channel"].astype(str)
            elif "FDS_Count" in tmp.columns and "RTL_Count" in tmp.columns:
                tmp["CustomerTypes"] = tmp.apply(lambda r: f"FDSx{int(r['FDS_Count'])}, RTLx{int(r['RTL_Count'])}", axis=1)
            else:
                tmp["CustomerTypes"] = ""

            # TargetSum_g isn't meaningful here; use Pounds_per_Run converted to grams for a proxy
            if "Pounds_per_Run" in tmp.columns:
                tmp["TargetSum_g"] = tmp["Pounds_per_Run"] * float(getattr(a3d3_core, "GRAMS_PER_LB", 453.592))
            else:
                tmp["TargetSum_g"] = None

            results = tmp

        return EngineResult(results_df=results, meta=meta, warnings=warnings)
