"""Backend API and inference checks for the final prediction service."""

import json
import threading
import unittest
import urllib.error
import urllib.request
import warnings
from http.server import ThreadingHTTPServer

from sklearn.exceptions import InconsistentVersionWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    from backend import app


class BackendServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.RequestHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def api_get(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=20) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def api_post(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def valid_payload(self):
        return {
            "hotel": "City Hotel",
            "arrival_date_year": 2026,
            "arrival_date_month": "July",
            "arrival_date_week_number": 28,
            "arrival_date_day_of_month": 12,
            "lead_time": 80,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "reserved_room_type": "A",
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "adr": 95,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
            "agent": "9",
            "company": "Not_recorded",
        }

    def test_health_and_schema_endpoints_report_loaded_model(self):
        status, health = self.api_get("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(health["model"], app.METADATA["model"])

        status, schema = self.api_get("/api/schema")
        self.assertEqual(status, 200)
        self.assertTrue(schema["ok"])
        self.assertEqual(schema["predictor_columns"], app.PREDICTOR_COLUMNS)
        self.assertIn("hotel", schema["options"])
        self.assertIn("lead_time", schema["int_ranges"])
        self.assertIn("adr", schema["float_ranges"])

    def test_prediction_endpoint_returns_meaningful_inference_result(self):
        status, result = self.api_post("/api/predict", self.valid_payload())
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertIn(result["prediction"], [0, 1])
        self.assertIn(result["prediction_label"], ["Likely to cancel", "Likely not to cancel"])
        self.assertIsInstance(result["cancellation_probability"], float)
        self.assertGreaterEqual(result["cancellation_probability"], 0.0)
        self.assertLessEqual(result["cancellation_probability"], 1.0)
        self.assertEqual(result["features"]["total_stay"], 3)
        self.assertEqual(result["features"]["total_guests"], 2)
        self.assertEqual(result["model"], app.METADATA["model"])

    def test_prediction_endpoint_rejects_missing_or_invalid_inputs(self):
        invalid = self.valid_payload()
        invalid["lead_time"] = -1
        invalid.pop("hotel")

        status, result = self.api_post("/api/predict", invalid)
        self.assertEqual(status, 400)
        self.assertFalse(result["ok"])
        self.assertIn("hotel", result["errors"])
        self.assertIn("lead_time", result["errors"])

        invalid = self.valid_payload()
        invalid["adults"] = 0
        invalid["children"] = 0
        invalid["babies"] = 0
        status, result = self.api_post("/api/predict", invalid)
        self.assertEqual(status, 400)
        self.assertFalse(result["ok"])
        self.assertIn("total_guests", result["errors"])


if __name__ == "__main__":
    unittest.main()
