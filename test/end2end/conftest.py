"""Pytest configuration for ovos-ddg-plugin tests.

Prevents any HiveMind network connections during the test session.
``HiveMessageBusClient.run_in_thread`` is a no-op here because the
HiveMind pipeline plugin is installed in this environment and
IntentService eagerly loads every registered pipeline plugin.
Without this guard the plugin tries to connect to a HiveMind node
and loops forever, hanging the whole test suite.
"""
from unittest.mock import patch


def pytest_configure(config):
    try:
        patch("hivemind_bus_client.HiveMessageBusClient.run_in_thread",
              return_value=None).start()
    except Exception:
        pass
