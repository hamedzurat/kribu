"""!
@file test_gui.py
@brief Unit tests for the Sholo Guti web GUI backend REST API.
"""

import json
import socketserver
import threading
import time
import urllib.request
import urllib.error

from kribu.gui import find_free_port, GUIHTTPRequestHandler


## @brief Helper to perform JSON POST requests to server.
def post_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


## @brief Verifies that the GUI HTTP REST server responds correctly.
def test_gui_api_endpoints() -> None:
    port = find_free_port()

    class TestServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = TestServer(("127.0.0.1", port), GUIHTTPRequestHandler)
    serverThread = threading.Thread(target=httpd.serve_forever)
    serverThread.daemon = True
    serverThread.start()

    time.sleep(0.3)

    try:
        # 1. Test HTTP GET 'engine_status'
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/engine_status")
        with urllib.request.urlopen(req) as response:
            statusData = json.loads(response.read().decode("utf-8"))
            assert "backend" in statusData
            assert "has_nn_model" in statusData

        # 2. Test HTTP POST 'init'
        initResp = post_json(f"http://127.0.0.1:{port}/api/init", {})
        assert "me" in initResp
        assert "opp" in initResp

        # 3. Test HTTP POST 'possible_moves'
        movesResp = post_json(
            f"http://127.0.0.1:{port}/api/possible_moves",
            {
                "me": initResp["me"],
                "opp": initResp["opp"],
                "activeCaptureIdx": initResp["activeCaptureIdx"],
                "currentPlayer": initResp["currentPlayer"],
            },
        )
        assert "moves" in movesResp
        assert len(movesResp["moves"]) > 0

    finally:
        httpd.shutdown()
        serverThread.join()
