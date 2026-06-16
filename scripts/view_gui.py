#!/usr/bin/env python3
"""
@file view_gui.py
@brief A lightweight backend web server to run the Sholo Guti web GUI.
@details Integrates the C++ nanobind engine rules, AI searches, and neural networks,
         and serves the custom HTML, CSS, and JS web page layout.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser

# Add path to the python package containing kribu modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_ROOT, "python", "src"))


def find_free_port(startPort: int = 8000, maxPort: int = 8100) -> int:
    """
    @brief Finds an available port on localhost to start the HTTP server.
    """
    import socket

    for port in range(startPort, maxPort):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"Could not find an available port in range {startPort}-{maxPort}.")


def open_browser_after_delay(url: str, delaySeconds: float = 0.6) -> None:
    """
    @brief Opens the browser to the specified URL after a brief delay.
    """
    time.sleep(delaySeconds)
    print(f"\n[info] Opening browser to {url} ...")
    webbrowser.open(url)


class GUIHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    @brief Request handler processing static files and custom board API endpoints.
    """

    def end_headers(self) -> None:
        """
        @brief Sends headers and includes cache-control headers to prevent browser caching.
        """
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        # Serve the home index file by default
        if self.path == "/" or self.path == "/index.html":
            self.path = "/gui/index.html"

        # Endpoint to verify backend engine status and trained NN models
        if self.path == "/api/engine_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            has_nn = os.path.exists(os.path.join(REPO_ROOT, "model.pt"))
            status = {
                "backend": "C++20 via nanobind",
                "has_nn_model": has_nn
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            params = json.loads(body) if body else {}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # Import kribu compiled module inside endpoints to avoid issues if not fully built yet
            import kribu

            response_data = {}

            # 1. Initialize Board
            if self.path == "/api/init":
                initial_state = kribu.INITIAL_STATE
                response_data = {
                    "me": str(initial_state.me),
                    "opp": str(initial_state.opp),
                    "activeCaptureIdx": int(initial_state.activeCaptureIdx),
                    "currentPlayer": "A"
                }

            # 2. Get Valid Possible Moves
            elif self.path == "/api/possible_moves":
                me = int(params["me"])
                opp = int(params["opp"])
                activeCaptureIdx = int(params["activeCaptureIdx"])
                currentPlayer = params["currentPlayer"]

                state = kribu.boardState()
                state.me = me
                state.opp = opp
                state.activeCaptureIdx = activeCaptureIdx

                # Flip board to match opponent's perspective if currentPlayer is B
                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    moves = kribu.all_possible_moves(flipped)
                else:
                    moves = kribu.all_possible_moves(state)

                decoded_moves = []
                for mId in moves:
                    if mId == 0:
                        decoded_moves.append({
                            "moveId": 0,
                            "from": -1,
                            "to": -1,
                            "captured": -1
                        })
                        continue

                    m = kribu.decode_move(mId)
                    decoded_moves.append({
                        "moveId": mId,
                        "from": int(m.fromNode),
                        "to": int(m.toNode),
                        "captured": int(m.captured)
                    })

                response_data = {"moves": decoded_moves}

            # 3. Apply Move Transitions
            elif self.path == "/api/apply_move":
                state_dict = params["state"]
                moveId = int(params["moveId"])
                currentPlayer = params["currentPlayer"]

                state = kribu.boardState()
                state.me = int(state_dict["me"])
                state.opp = int(state_dict["opp"])
                state.activeCaptureIdx = int(state_dict["activeCaptureIdx"])

                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    next_flipped = kribu.apply_move(flipped, moveId)
                    next_state = kribu.flip_board(next_flipped)
                else:
                    next_state = kribu.apply_move(state, moveId)

                response_data = {
                    "nextState": {
                        "me": str(next_state.me),
                        "opp": str(next_state.opp),
                        "activeCaptureIdx": int(next_state.activeCaptureIdx)
                    }
                }

            # 4. Fetch Game Outcomes
            elif self.path == "/api/game_status":
                state = kribu.boardState()
                state.me = int(params["me"])
                state.opp = int(params["opp"])
                state.activeCaptureIdx = int(params["activeCaptureIdx"])
                currentPlayer = params["currentPlayer"]

                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    status = kribu.get_game_status(flipped)
                    # Convert to B relative string representation
                    if status == kribu.GameStatus.ME_WINS:
                        status_str = "OPP_WINS"
                    elif status == kribu.GameStatus.OPP_WINS:
                        status_str = "ME_WINS"
                    else:
                        status_str = "ONGOING"
                else:
                    status = kribu.get_game_status(state)
                    status_str = status.name

                response_data = {"status": status_str}

            # 5. AI Calculations (Minimax Search / NN inference)
            elif self.path == "/api/ai_move":
                state_dict = params["state"]
                currentPlayer = params["currentPlayer"]
                ai_type = params["aiType"]
                depth = int(params["depth"])

                state = kribu.boardState()
                state.me = int(state_dict["me"])
                state.opp = int(state_dict["opp"])
                state.activeCaptureIdx = int(state_dict["activeCaptureIdx"])

                if currentPlayer == "B":
                    active_state = kribu.flip_board(state)
                else:
                    active_state = state

                best_move_id = -1
                if ai_type == "nn":
                    global nn_model
                    if 'nn_model' not in globals():
                        import torch
                        from kribu.train import SholoGutiNet
                        nn_model = SholoGutiNet()
                        model_path = os.path.join(REPO_ROOT, "model.pt")
                        nn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
                        nn_model.eval()

                    import torch
                    from kribu.train import prepare_state_input
                    inp = prepare_state_input(active_state)
                    with torch.no_grad():
                        policy_logits, _ = nn_model(inp)

                    valid_moves = kribu.all_possible_moves(active_state)
                    if valid_moves:
                        mask = torch.full_like(policy_logits[0], float("-inf"))
                        for mId in valid_moves:
                            mask[mId] = 0.0
                        masked_logits = policy_logits[0] + mask
                        best_move_id = int(torch.argmax(masked_logits).item())
                else:
                    res = kribu.minimax(active_state, depth)
                    best_move_id = res.moveId

                if best_move_id != -1:
                    next_active = kribu.apply_move(active_state, best_move_id)
                    if currentPlayer == "B":
                        next_state = kribu.flip_board(next_active)
                    else:
                        next_state = next_active

                    if best_move_id == 0:
                        move_dict = {"from": -1, "to": -1, "captured": -1}
                    else:
                        m = kribu.decode_move(best_move_id)
                        move_dict = {
                            "from": int(m.fromNode),
                            "to": int(m.toNode),
                            "captured": int(m.captured)
                        }
                else:
                    next_state = state
                    move_dict = {"from": -1, "to": -1, "captured": -1}

                response_data = {
                    "moveId": best_move_id,
                    "move": move_dict,
                    "nextState": {
                        "me": str(next_state.me),
                        "opp": str(next_state.opp),
                        "activeCaptureIdx": int(next_state.activeCaptureIdx)
                    }
                }

            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_error(404, "Not Found")


def main() -> None:
    # Always serve from the repository root
    os.chdir(REPO_ROOT)

    try:
        port = find_free_port()
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}/index.html"

    # Start browser helper thread
    browserThread = threading.Thread(target=open_browser_after_delay, args=(url,))
    browserThread.daemon = True
    browserThread.start()

    Handler = GUIHTTPRequestHandler
    Handler.log_message = lambda self, format, *args: None

    print("=" * 60)
    print("   KRIBU SHOLO GUTI WEB FRONTEND SERVER")
    print("=" * 60)
    print(f"  Serving game board at: {url}")
    print("  Style sheet loaded:   doc/doxygen-awesome.css")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Web server stopped. Exiting.")
    except Exception as e:
        print(f"\n[error] Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
