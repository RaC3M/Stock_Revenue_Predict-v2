from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hybrid_forecast import export_forecast
from hybrid_forecast.live_engine import CoreForecastResult, LiveForecastResult


class ExportForecastTests(unittest.TestCase):
    def _core(self) -> CoreForecastResult:
        return CoreForecastResult(
            monthly=pd.DataFrame([{"stock_id": 1}]),
            summary=pd.DataFrame([{"as_of_date": "2026-09-19"}]),
            quarterly_eps=pd.DataFrame(), payout_history=pd.DataFrame(),
            data_status=pd.DataFrame(), order_search=pd.DataFrame(), notes=[],
            artifact_id="artifact", generated_at="2026-09-19T00:00:00+00:00",
        )

    def _live(self) -> LiveForecastResult:
        core = self._core()
        return LiveForecastResult(
            monthly=core.monthly, summary=core.summary,
            quarterly_eps=core.quarterly_eps, payout_history=core.payout_history,
            data_status=core.data_status, order_search=core.order_search, notes=[],
            artifact_id=core.artifact_id, generated_at=core.generated_at,
            price_source="observed_csv",
        )

    def test_batch_continues_after_one_stock_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            def load_core(request):
                if request.stock_id == 2:
                    raise ValueError("expected failure")
                return self._core()

            argv = [
                "export_forecast", "--data-dir", tmp, "--as-of", "2026-09-19",
                "--stock", "1", "--stock", "2", "--output-dir", tmp,
            ]
            output = io.StringIO()
            with (
                patch("sys.argv", argv),
                patch.object(export_forecast, "load_or_build_forecast", side_effect=load_core),
                patch.object(export_forecast, "load_latest_price", return_value=(100.0, pd.Timestamp("2026-09-19"))),
                patch.object(export_forecast, "apply_price_scenario", return_value=self._live()),
                patch.object(export_forecast, "write_llm_bundle", return_value=Path(tmp) / "bundle"),
                redirect_stdout(output),
            ):
                code = export_forecast.main()
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual([row["stock_id"] for row in payload["completed"]], [1])
            self.assertEqual([row["stock_id"] for row in payload["failures"]], [2])


if __name__ == "__main__":
    unittest.main()
