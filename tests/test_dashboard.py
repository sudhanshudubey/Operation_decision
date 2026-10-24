"""Exercise the shipped dataset through the actual Streamlit script."""

import unittest

from streamlit.testing.v1 import AppTest

from retail.data import ROOT


class DashboardTests(unittest.TestCase):
    def test_every_available_product_and_sensitivity_mode(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        product = next(item for item in app.selectbox if item.label == "Product")
        for i in range(len(product.options)):
            with self.subTest(product=i):
                next(item for item in app.selectbox if item.label == "Product").select_index(i).run()
                self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
                self.assertTrue(any(item.label == "Final test MAE" for item in app.metric))
        next(item for item in app.selectbox if item.label == "Sales interpretation").select_index(1).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        next(item for item in app.slider if item.label == "Lead time (days)").set_value(7).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
