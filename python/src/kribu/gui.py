"""!
@file gui.py
@brief A robust backend web server supporting static files and WebSocket communication.
@details Integrates search players, PyTorch neural networks, and standard WebSocket framing.
"""

import base64
import hashlib
import http.server
import json
import os
import random
import socketserver
import struct
import sys
import threading
import time
import webbrowser

# Locate the repository root dynamically
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Expose local python source path for imports
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "src"))
import kribu
from arena.player import NeuralPlayer


## @brief Finds an available port on localhost to start the HTTP server.
#  @param startPort The starting port (default 8000).
#  @param maxPort The maximum port to search (default 8100).
#  @return An available port number.
def find_free_port(startPort: int = 8000, maxPort: int = 8100) -> int:
    import socket

    for port in range(startPort, maxPort):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"Could not find an available port in range {startPort}-{maxPort}.")


## @brief Opens the browser to the specified URL after a brief delay.
#  @param url The URL to open.
#  @param delaySeconds The delay in seconds before opening (default 0.6).
def open_browser_after_delay(url: str, delaySeconds: float = 0.6) -> None:
    time.sleep(delaySeconds)
    print(f"\n[info] Opening browser to {url} ...")
    webbrowser.open(url)


## @brief Global cache for the neural network player.
nnPlayerInstance = None
nnPlayerModelPath = ""


## @brief Helper to load or retrieve the Neural Player instance.
#  @param modelPath Path to the PyTorch model checkpoint.
#  @return NeuralPlayer instance.
def get_neural_player(modelPath: str) -> NeuralPlayer:
    global nnPlayerInstance, nnPlayerModelPath
    if nnPlayerInstance is None or nnPlayerModelPath != modelPath:
        nnPlayerInstance = NeuralPlayer(model_path=modelPath)
        nnPlayerModelPath = modelPath
    return nnPlayerInstance


## @brief Request handler processing static files and custom board API endpoints.
class GUIHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve relative to the parent directory of this file
        currentDir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=currentDir, **kwargs)

    def end_headers(self) -> None:
        """!
        @brief Sends headers and includes cache-control headers to prevent browser caching.
        """
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        """!
        @brief Override GET method to serve index.html or handle engine status.
        """
        if self.path == "/api/engine_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            modelsDir = os.path.join(REPO_ROOT, "models")
            availableModels = []
            if os.path.exists(modelsDir):
                availableModels = [f for f in os.listdir(modelsDir) if f.endswith(".pt")]
            responseData = {
                "backend": "C++20 via nanobind",
                "available_models": availableModels,
                "has_nn_model": len(availableModels) > 0,
            }
            self.wfile.write(json.dumps(responseData).encode("utf-8"))
            return

        # Serve the home index file by default
        if self.path == "/" or self.path == "/index.html":
            self.path = "/gui/index.html"

        return super().do_GET()

    def do_POST(self) -> None:
        """!
        @brief Handles POST API requests for Sholo Guti engine.
        """
        if self.path.startswith("/api/"):
            contentLength = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(contentLength).decode("utf-8")
            params = json.loads(body) if body else {}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            action = self.path.split("/")[-1]
            responseData = {}

            # 1. Initialize Board
            if action == "init":
                initialState = kribu.INITIAL_STATE
                responseData = {
                    "me": str(initialState.me),
                    "opp": str(initialState.opp),
                    "activeCaptureIdx": int(initialState.activeCaptureIdx),
                    "currentPlayer": "A",
                }

            # 2. Get Valid Possible Moves
            elif action == "possible_moves":
                meVal = int(params["me"])
                oppVal = int(params["opp"])
                activeCaptureIdx = int(params["activeCaptureIdx"])
                currentPlayer = params["currentPlayer"]

                state = kribu.boardState()
                state.me = meVal
                state.opp = oppVal
                state.activeCaptureIdx = activeCaptureIdx

                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    moves = kribu.all_possible_moves(flipped)
                else:
                    moves = kribu.all_possible_moves(state)

                decodedMoves = []
                for mId in moves:
                    if mId == 0:
                        decodedMoves.append({"moveId": 0, "from": -1, "to": -1, "captured": -1})
                        continue

                    m = kribu.decode_move(mId)
                    decodedMoves.append(
                        {
                            "moveId": mId,
                            "from": int(m.fromNode),
                            "to": int(m.toNode),
                            "captured": int(m.captured),
                        }
                    )

                responseData = {"moves": decodedMoves}

            # 3. Apply Move Transitions
            elif action == "apply_move":
                stateDict = params["state"]
                moveId = int(params["moveId"])
                currentPlayer = params["currentPlayer"]

                state = kribu.boardState()
                state.me = int(stateDict["me"])
                state.opp = int(stateDict["opp"])
                state.activeCaptureIdx = int(stateDict["activeCaptureIdx"])

                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    nextFlipped = kribu.apply_move(flipped, moveId)
                    nextState = kribu.flip_board(nextFlipped)
                else:
                    nextState = kribu.apply_move(state, moveId)

                responseData = {
                    "nextState": {
                        "me": str(nextState.me),
                        "opp": str(nextState.opp),
                        "activeCaptureIdx": int(nextState.activeCaptureIdx),
                    }
                }

            # 4. Fetch Game Outcomes
            elif action == "game_status":
                state = kribu.boardState()
                state.me = int(params["me"])
                state.opp = int(params["opp"])
                state.activeCaptureIdx = int(params["activeCaptureIdx"])
                currentPlayer = params["currentPlayer"]

                if currentPlayer == "B":
                    flipped = kribu.flip_board(state)
                    status = kribu.get_game_status(flipped)
                    if status == kribu.GameStatus.ME_WINS_ELIMINATION or status == kribu.GameStatus.ME_WINS_STALEMATE:
                        statusStr = "OPP_WINS"
                    elif (
                        status == kribu.GameStatus.OPP_WINS_ELIMINATION or status == kribu.GameStatus.OPP_WINS_STALEMATE
                    ):
                        statusStr = "ME_WINS"
                    else:
                        statusStr = "ONGOING"
                else:
                    status = kribu.get_game_status(state)
                    if status == kribu.GameStatus.ME_WINS_ELIMINATION or status == kribu.GameStatus.ME_WINS_STALEMATE:
                        statusStr = "ME_WINS"
                    elif (
                        status == kribu.GameStatus.OPP_WINS_ELIMINATION or status == kribu.GameStatus.OPP_WINS_STALEMATE
                    ):
                        statusStr = "OPP_WINS"
                    else:
                        statusStr = "ONGOING"

                responseData = {"status": statusStr}

            # 5. AI Calculations
            elif action == "ai_move":
                stateDict = params["state"]
                currentPlayer = params["currentPlayer"]
                aiType = params.get("aiType", "minimax")
                depth = int(params.get("depth", 5))
                modelFile = params.get("modelFile", "search_blend_joint.pt")

                state = kribu.boardState()
                state.me = int(stateDict["me"])
                state.opp = int(stateDict["opp"])
                state.activeCaptureIdx = int(stateDict["activeCaptureIdx"])

                if currentPlayer == "B":
                    activeState = kribu.flip_board(state)
                else:
                    activeState = state

                bestMoveId = -1
                validMoves = list(kribu.all_possible_moves(activeState))

                if validMoves:
                    if aiType == "nn":
                        modelPath = os.path.join(REPO_ROOT, "models", modelFile)
                        if not os.path.exists(modelPath):
                            modelsDir = os.path.join(REPO_ROOT, "models")
                            ptFiles = [f for f in os.listdir(modelsDir) if f.endswith(".pt")]
                            if ptFiles:
                                modelPath = os.path.join(modelsDir, ptFiles[0])

                        if os.path.exists(modelPath):
                            player = get_neural_player(modelPath)
                            bestMoveId, _ = player.get_move(
                                activeState.me,
                                activeState.opp,
                                activeState.activeCaptureIdx,
                                valid_moves=validMoves,
                                state=activeState,
                            )
                        else:
                            bestMoveId = validMoves[0]
                    elif aiType == "minimax":
                        res = kribu.minimax(activeState, depth)
                        bestMoveId = res.moveId
                    elif aiType == "minimax_player_8":
                        bestMoveId = kribu.minimax_player_8(activeState)
                    elif aiType == "minimax_player_4":
                        bestMoveId = kribu.minimax_player_4(activeState)
                    elif aiType == "minimax_player_8_mad2":
                        bestMoveId = kribu.minimax_player_8_mad2(activeState)
                    elif aiType == "mcts_player_800":
                        bestMoveId = kribu.mcts_player_800(activeState)
                    elif aiType == "greedy_player":
                        bestMoveId = kribu.greedy_player(activeState)
                    elif aiType == "random":
                        bestMoveId = random.choice(validMoves)
                    else:
                        res = kribu.minimax(activeState, depth)
                        bestMoveId = res.moveId

                if bestMoveId != -1:
                    nextActive = kribu.apply_move(activeState, bestMoveId)
                    if currentPlayer == "B":
                        nextState = kribu.flip_board(nextActive)
                    else:
                        nextState = nextActive

                    if bestMoveId == 0:
                        moveDict = {"from": -1, "to": -1, "captured": -1}
                    else:
                        m = kribu.decode_move(bestMoveId)
                        moveDict = {
                            "from": int(m.fromNode),
                            "to": int(m.toNode),
                            "captured": int(m.captured),
                        }
                else:
                    nextState = state
                    moveDict = {"from": -1, "to": -1, "captured": -1}

                responseData = {
                    "moveId": bestMoveId,
                    "move": moveDict,
                    "nextState": {
                        "me": str(nextState.me),
                        "opp": str(nextState.opp),
                        "activeCaptureIdx": int(nextState.activeCaptureIdx),
                    },
                }

            self.wfile.write(json.dumps(responseData).encode("utf-8"))
            return

        self.send_error(404, "Not Found")


## @brief Custom TCP Server that sets SO_LINGER and SO_REUSEADDR for instant port freeing.
class NeoTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

    def server_bind(self) -> None:
        import socket
        import struct

        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        super().server_bind()


## @brief Runs the web GUI server.
#  @param openBrowser If True, opens the web browser to the server URL automatically (default False).
def run_gui(openBrowser: bool = False) -> None:
    import signal

    def handle_sigint(sig, frame):
        print("\n[info] Web server stopped. Exiting.")
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    port = 8000
    url = f"http://127.0.0.1:{port}/index.html"

    # Start browser helper thread if requested
    if openBrowser:
        browserThread = threading.Thread(target=open_browser_after_delay, args=(url,))
        browserThread.daemon = True
        browserThread.start()

    Handler = GUIHTTPRequestHandler
    Handler.log_message = lambda self, format, *args: None

    print("=" * 60)
    print("   KRIBU SHOLO GUTI WEB FRONTEND SERVER (WebSocket)")
    print("=" * 60)
    print(f"  Serving game board at: {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        with NeoTCPServer(("127.0.0.1", port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Web server stopped. Exiting.")
    except Exception as e:
        print(f"\n[error] Server error: {e}", file=sys.stderr)
        sys.exit(1)
