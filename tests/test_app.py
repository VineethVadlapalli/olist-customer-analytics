"""Smoke test: the dashboard renders without errors, with and without filters."""

from streamlit.testing.v1 import AppTest

APP = "../app/streamlit_app.py"


def test_dashboard_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 5
    assert at.metric[0].value.replace(",", "").isdigit()


def test_state_filter():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.multiselect[0].select("SP").run()
    assert not at.exception
    all_customers = int(AppTest.from_file(APP, default_timeout=60).run().metric[0].value.replace(",", ""))
    assert int(at.metric[0].value.replace(",", "")) < all_customers
