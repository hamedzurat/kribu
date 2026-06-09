// Kribu Sholo Guti - Premium Web Frontend JavaScript Logic

// Define node coordinates mapped symmetrically
const nodeCoordinates = {
    // Top pyramid (Row -2 and -1)
    0: { x: -160, y: -320 },
    1: { x: 0, y: -320 },
    2: { x: 160, y: -320 },
    3: { x: -80, y: -240 },
    4: { x: 0, y: -240 },
    5: { x: 80, y: -240 },
    
    // Central 5x5 Grid
    // Row 0 (y = -160)
    6: { x: -160, y: -160 },
    7: { x: -80, y: -160 },
    8: { x: 0, y: -160 },
    9: { x: 80, y: -160 },
    10: { x: 160, y: -160 },
    
    // Row 1 (y = -80)
    11: { x: -160, y: -80 },
    12: { x: -80, y: -80 },
    13: { x: 0, y: -80 },
    14: { x: 80, y: -80 },
    15: { x: 160, y: -80 },
    
    // Row 2 (y = 0)
    16: { x: -160, y: 0 },
    17: { x: -80, y: 0 },
    18: { x: 0, y: 0 },
    19: { x: 80, y: 0 },
    20: { x: 160, y: 0 },
    
    // Row 3 (y = 80)
    21: { x: -160, y: 80 },
    22: { x: -80, y: 80 },
    23: { x: 0, y: 80 },
    24: { x: 80, y: 80 },
    25: { x: 160, y: 80 },
    
    // Row 4 (y = 160)
    26: { x: -160, y: 160 },
    27: { x: -80, y: 160 },
    28: { x: 0, y: 160 },
    29: { x: 80, y: 160 },
    30: { x: 160, y: 160 },
    
    // Bottom pyramid (Row 5 and 6)
    31: { x: -80, y: 240 },
    32: { x: 0, y: 240 },
    33: { x: 80, y: 240 },
    34: { x: -160, y: 320 },
    35: { x: 0, y: 320 },
    36: { x: 160, y: 320 }
};

// Structural lines of the board graph
const boardSegments = [
    // Horizontals
    [0, 1], [1, 2], [3, 4], [4, 5],
    [6, 7], [7, 8], [8, 9], [9, 10],
    [11, 12], [12, 13], [13, 14], [14, 15],
    [16, 17], [17, 18], [18, 19], [19, 20],
    [21, 22], [22, 23], [23, 24], [24, 25],
    [26, 27], [27, 28], [28, 29], [29, 30],
    [31, 32], [32, 33], [34, 35], [35, 36],
    
    // Verticals
    [1, 4], [4, 8],
    [6, 11], [11, 16], [16, 21], [21, 26],
    [7, 12], [12, 17], [17, 22], [22, 27],
    [8, 13], [13, 18], [18, 23], [23, 28],
    [9, 14], [14, 19], [19, 24], [24, 29],
    [10, 15], [15, 20], [20, 25], [25, 30],
    [28, 32], [32, 35],
    
    // Diagonals
    [0, 3], [3, 8],
    [2, 5], [5, 8],
    [6, 12], [12, 18],
    [8, 12], [12, 16],
    [8, 14], [14, 20],
    [10, 14], [14, 18],
    [16, 22], [22, 28],
    [18, 22], [22, 26],
    [18, 24], [24, 30],
    [20, 24], [24, 28],
    [28, 31], [31, 34],
    [28, 33], [33, 36]
];

// Game State Class
class GameState {
    constructor() {
        this.me = "0";                 // Active player piece mask (string or bigint)
        this.opp = "0";                // Opponent piece mask (string or bigint)
        this.activeCaptureIdx = -1;    // Locked piece index for multi-captures (-1 if none)
        this.currentPlayer = 'A';      // Current turn: 'A' or 'B'
        this.gameMode = 'pve';         // Mode: 'pve', 'pve_nn', 'pvp', 'eve'
        this.userRole = 'A';           // Human role: 'A' (Green), 'B' (Red), or none
        this.aiDepth = 5;
        this.gameStatus = 'ONGOING';   // 'ONGOING', 'ME_WINS', 'OPP_WINS'
        
        this.possibleMoves = [];       // Array of decoded moves for the active player
        this.selectedPiece = null;     // Selected node index
        this.validDestinations = [];   // Node indices that are valid targets for selectedPiece
        this.history = [];             // Stack of states for undo
        this.moveLog = [];             // Text records of moves
        this.moveSnapshots = [];       // Board snapshots for each logged move
        this.viewingHistoryIndex = -1; // -1 = live game, >= 0 = viewing past state
        this.isAnimating = false;      // Block input during animations
    }

    // Reset state to initial game configuration
    reset() {
        this.me = "0";
        this.opp = "0";
        this.activeCaptureIdx = -1;
        this.currentPlayer = 'A';
        this.gameStatus = 'ONGOING';
        this.possibleMoves = [];
        this.selectedPiece = null;
        this.validDestinations = [];
        this.history = [];
        this.moveLog = [];
        this.moveSnapshots = [];
        this.viewingHistoryIndex = -1;
        this.isAnimating = false;
    }

    // Set mask using active player & opponent values
    setMasks(me, opp, activeCaptureIdx) {
        this.me = me;
        this.opp = opp;
        this.activeCaptureIdx = activeCaptureIdx;
    }

    // Push current state to undo history
    pushHistory() {
        this.history.push({
            me: this.me,
            opp: this.opp,
            activeCaptureIdx: this.activeCaptureIdx,
            currentPlayer: this.currentPlayer,
            gameStatus: this.gameStatus,
            moveLog: [...this.moveLog]
        });
        document.getElementById('btn-undo').disabled = false;
    }

    // Pop the previous state
    popHistory() {
        if (this.history.length === 0) return;
        const prevState = this.history.pop();
        this.me = prevState.me;
        this.opp = prevState.opp;
        this.activeCaptureIdx = prevState.activeCaptureIdx;
        this.currentPlayer = prevState.currentPlayer;
        this.gameStatus = prevState.gameStatus;
        this.moveLog = prevState.moveLog;
        this.selectedPiece = null;
        this.validDestinations = [];
        
        if (this.history.length === 0) {
            document.getElementById('btn-undo').disabled = true;
        }
    }
}

const game = new GameState();

// Initialize application on load
document.addEventListener("DOMContentLoaded", () => {
    setupBoardLines();
    initializeSVG();
    setupEventListeners();
    checkEngineStatus();
    startNewGame();
});

// Determine whether the board layout should be flipped
function shouldFlipBoard() {
    return false; // The user explicitly requested NO flipping ever.
}

// Retrieve coordinates for a given node index
function getCoordinate(nodeId) {
    return nodeCoordinates[nodeId];
}

// Draw background grid lines on SVG
function setupBoardLines() {
    const linesGroup = document.getElementById('board-lines');
    linesGroup.innerHTML = '';
    
    boardSegments.forEach(([from, to]) => {
        const fromCoord = getCoordinate(from);
        const toCoord = getCoordinate(to);
        
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", fromCoord.x);
        line.setAttribute("y1", fromCoord.y);
        line.setAttribute("x2", toCoord.x);
        line.setAttribute("y2", toCoord.y);
        line.setAttribute("class", "board-line");
        line.setAttribute("id", `line-${from}-${to}`);
        linesGroup.appendChild(line);
    });
}

// Bind all UI interactive controls
function setupEventListeners() {
    // Mode selectors
    document.getElementById('game-mode').addEventListener('change', (e) => {
        game.gameMode = e.target.value;
        const aiDepthGroup = document.getElementById('ai-options-group');
        const roleGroup = document.getElementById('player-role-group');
        
        if (game.gameMode === 'pve') {
            aiDepthGroup.style.display = 'block';
            roleGroup.style.display = 'block';
        } else if (game.gameMode === 'pve_nn') {
            aiDepthGroup.style.display = 'none';
            roleGroup.style.display = 'block';
        } else if (game.gameMode === 'pvp') {
            aiDepthGroup.style.display = 'none';
            roleGroup.style.display = 'none';
        } else if (game.gameMode === 'eve') {
            aiDepthGroup.style.display = 'block';
            roleGroup.style.display = 'none';
        }
    });

    // AI Depth slider
    const depthInput = document.getElementById('ai-depth');
    const depthVal = document.getElementById('depth-value');
    depthInput.addEventListener('input', (e) => {
        depthVal.textContent = e.target.value;
        game.aiDepth = parseInt(e.target.value, 10);
    });

    // Role switcher (Green/Red)
    document.getElementById('btn-play-green').addEventListener('click', () => {
        document.getElementById('btn-play-green').classList.add('active');
        document.getElementById('btn-play-red').classList.remove('active');
        game.userRole = 'A';
    });

    document.getElementById('btn-play-red').addEventListener('click', () => {
        document.getElementById('btn-play-red').classList.add('active');
        document.getElementById('btn-play-green').classList.remove('active');
        game.userRole = 'B';
    });

    // Action buttons
    document.getElementById('btn-new-game').addEventListener('click', startNewGame);
    document.getElementById('btn-undo').addEventListener('click', handleUndo);
    document.getElementById('btn-end-chain').addEventListener('click', handleEndChain);
    document.getElementById('btn-return-current').addEventListener('click', returnToCurrentGame);
    
    // Modal buttons
    document.getElementById('btn-modal-restart').addEventListener('click', () => {
        document.getElementById('game-over-modal').style.display = 'none';
        startNewGame();
    });
    
    document.getElementById('btn-modal-close').addEventListener('click', () => {
        document.getElementById('game-over-modal').style.display = 'none';
    });

    // Theme Switcher Toggle
    document.getElementById('theme-toggle').addEventListener('click', () => {
        const html = document.documentElement;
        if (html.classList.contains('dark-mode')) {
            html.classList.remove('dark-mode');
            html.classList.add('light-mode');
        } else {
            html.classList.remove('light-mode');
            html.classList.add('dark-mode');
        }
    });
}

// Check Backend Engine Status & NN availability
async function checkEngineStatus() {
    try {
        const response = await fetch('/api/engine_status');
        const data = await response.json();
        
        document.getElementById('metric-backend').textContent = data.backend || 'C++20';
        const nnStatus = document.getElementById('metric-nn-status');
        if (data.has_nn_model) {
            nnStatus.textContent = 'Loaded (model.pt)';
            nnStatus.style.color = 'var(--color-green)';
        } else {
            nnStatus.textContent = 'Unavailable';
            nnStatus.style.color = 'var(--app-text-muted)';
            // Disable neural net option if model is missing
            const select = document.getElementById('game-mode');
            const nnOpt = select.querySelector('option[value="pve_nn"]');
            if (nnOpt) {
                nnOpt.disabled = true;
                nnOpt.textContent = 'Player vs AI (NN - model.pt missing)';
            }
        }
    } catch (error) {
        console.error('Failed to get engine status:', error);
        document.getElementById('metric-nn-status').textContent = 'Error';
    }
}

// Start a fresh game configuration
async function startNewGame() {
    set_loading(true);
    game.reset();
    
    // Re-initialize SVG coordinates based on the selected player role/mode
    setupBoardLines();
    initializeSVG();
    
    try {
        const response = await fetch('/api/init', { method: 'POST' });
        const data = await response.json();
        
        game.setMasks(data.me, data.opp, data.activeCaptureIdx);
        game.currentPlayer = data.currentPlayer;
        
        updateUI();
        await fetchPossibleMoves();
        
        // If mode is EvE (AI vs AI), trigger the first AI move
        if (game.gameMode === 'eve') {
            setTimeout(triggerAIMove, 600);
        } else if (game.gameMode.startsWith('pve') && game.currentPlayer !== game.userRole) {
            // Player vs AI, and AI goes first (Player A is AI, Player role B is Human)
            setTimeout(triggerAIMove, 600);
        }
    } catch (e) {
        console.error('Error starting game:', e);
        alert('Could not start a new game. Make sure the server is running.');
    } finally {
        set_loading(false);
    }
}

// Fetch possible moves for current player
async function fetchPossibleMoves() {
    try {
        const response = await fetch('/api/possible_moves', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                me: game.me,
                opp: game.opp,
                activeCaptureIdx: game.activeCaptureIdx,
                currentPlayer: game.currentPlayer
            })
        });
        const data = await response.json();
        game.possibleMoves = data.moves;
    } catch (e) {
        console.error('Failed to fetch possible moves:', e);
    }
}

// Update the entire HTML / SVG UI based on state
function updateUI() {
    renderCounts();
    renderBoard();
    renderHistory();
    updateStatusText();
    
    // Toggle active capture chain button
    const endChainBtn = document.getElementById('btn-end-chain');
    if (game.activeCaptureIdx !== -1 && isHumanTurn()) {
        endChainBtn.style.display = 'block';
    } else {
        endChainBtn.style.display = 'none';
    }
}

// Render piece counts
function renderCounts() {
    let piecesA = 0;
    let piecesB = 0;
    
    // Decode bits in board masks
    // 'me' and 'opp' are binary string masks of length 37 (bits 0-36)
    // In our system, if currentPlayer is A:
    // 'me' is Player A pieces, 'opp' is Player B pieces.
    // If currentPlayer is B:
    // 'me' is Player B pieces, 'opp' is Player A pieces.
    let maskA = "0";
    let maskB = "0";
    if (game.currentPlayer === 'A') {
        maskA = game.me;
        maskB = game.opp;
    } else {
        maskA = game.opp;
        maskB = game.me;
    }
    
    for (let i = 0; i < 37; i++) {
        if (getBit(maskA, i)) piecesA++;
        if (getBit(maskB, i)) piecesB++;
    }
    
    document.getElementById('count-a').textContent = `${piecesA} Pieces`;
    document.getElementById('count-b').textContent = `${piecesB} Pieces`;
}

// Update status bar header
function updateStatusText() {
    const badge = document.getElementById('game-status-text');
    
    if (game.gameStatus !== 'ONGOING') {
        badge.classList.remove('pulse');
        badge.style.backgroundColor = 'var(--color-yellow)';
        badge.textContent = game.gameStatus === 'ME_WINS' ? 'PLAYER A WINS!' : 'PLAYER B WINS!';
        showGameOverModal();
        return;
    }
    
    if (isHumanTurn()) {
        badge.textContent = "Your Turn";
        badge.style.backgroundColor = game.currentPlayer === 'A' ? 'var(--color-green)' : 'var(--color-red)';
        badge.style.boxShadow = `0 0 10px ${game.currentPlayer === 'A' ? 'var(--color-green)' : 'var(--color-red)'}`;
    } else {
        badge.textContent = "AI is Thinking...";
        badge.style.backgroundColor = 'var(--primary-color)';
    }
}

// Check if it is a human player's turn to make a move
function isHumanTurn() {
    if (game.gameMode === 'pvp') return true;
    if (game.gameMode === 'eve') return false;
    return game.currentPlayer === game.userRole;
}

// Pre-initialize board elements once
function initializeSVG() {
    const nodesGroup = document.getElementById('board-nodes');
    const piecesGroup = document.getElementById('board-pieces');
    const highlightsGroup = document.getElementById('board-highlights');
    
    nodesGroup.innerHTML = '';
    piecesGroup.innerHTML = '';
    highlightsGroup.innerHTML = '';
    
    for (let i = 0; i < 37; i++) {
        const coord = getCoordinate(i);
        
        // 1. Draw SVG nodes (empty locations)
        const nodeCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        nodeCircle.setAttribute("cx", coord.x);
        nodeCircle.setAttribute("cy", coord.y);
        nodeCircle.setAttribute("r", 9);
        nodeCircle.setAttribute("class", "board-node-circle");
        nodeCircle.setAttribute("id", `node-${i}`);
        nodeCircle.addEventListener('click', () => handleNodeClick(i));
        nodesGroup.appendChild(nodeCircle);
        
        // Add subtle index label inside empty nodes
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", coord.x);
        label.setAttribute("y", coord.y);
        label.setAttribute("class", "node-label");
        label.setAttribute("id", `node-label-${i}`);
        nodesGroup.appendChild(label);
        
        // 2. Draw piece groups
        const pieceG = document.createElementNS("http://www.w3.org/2000/svg", "g");
        pieceG.setAttribute("id", `piece-group-${i}`);
        pieceG.style.display = 'none'; // hidden by default
        
        const pieceCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        pieceCircle.setAttribute("cx", coord.x);
        pieceCircle.setAttribute("cy", coord.y);
        pieceCircle.setAttribute("r", 14);
        pieceG.appendChild(pieceCircle);
        
        // Premium design details inside the piece (inner design ring)
        const innerRing = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        innerRing.setAttribute("cx", coord.x);
        innerRing.setAttribute("cy", coord.y);
        innerRing.setAttribute("r", 9);
        innerRing.setAttribute("fill", "none");
        innerRing.setAttribute("stroke", "rgba(255,255,255,0.4)");
        innerRing.setAttribute("stroke-width", "1");
        pieceG.appendChild(innerRing);
        
        // Put index label inside the piece
        const pieceText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        pieceText.setAttribute("x", coord.x);
        pieceText.setAttribute("y", coord.y);
        pieceText.setAttribute("fill", "#fff");
        pieceText.setAttribute("font-size", "9px");
        pieceText.setAttribute("font-weight", "800");
        pieceText.setAttribute("text-anchor", "middle");
        pieceText.setAttribute("dominant-baseline", "middle");
        pieceText.setAttribute("id", `piece-text-${i}`);
        pieceG.appendChild(pieceText);
        
        pieceG.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent node circle triggering
            handlePieceClick(i);
        });
        
        piecesGroup.appendChild(pieceG);

        // 3. Draw a static overlay circle for move targets
        const targetPulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        targetPulse.setAttribute("cx", coord.x);
        targetPulse.setAttribute("cy", coord.y);
        targetPulse.setAttribute("r", 17);
        targetPulse.setAttribute("fill", "none");
        targetPulse.setAttribute("stroke", "var(--color-yellow)");
        targetPulse.setAttribute("stroke-width", "2");
        targetPulse.setAttribute("stroke-dasharray", "3,3");
        targetPulse.setAttribute("id", `target-pulse-${i}`);
        targetPulse.style.display = 'none';
        highlightsGroup.appendChild(targetPulse);
    }
}

// Render nodes and pieces inside SVG (optimized incremental rendering)
function renderBoard() {
    // Parse masks as BigInt once to avoid performance overhead in loops
    const meMask = BigInt(game.me);
    const oppMask = BigInt(game.opp);
    
    const activePlayerA = game.currentPlayer === 'A';
    
    // Find absolute activeCaptureIdx
    let absoluteCaptureIdx = game.activeCaptureIdx;
    
    for (let i = 0; i < 37; i++) {
        const hasA = (meMask & (1n << BigInt(i))) !== 0n;
        const hasB = (oppMask & (1n << BigInt(i))) !== 0n;
        
        const displayNodeIndex = i;
        
        // 1. Update SVG node circles
        const nodeCircle = document.getElementById(`node-${i}`);
        nodeCircle.className.baseVal = "board-node-circle";
        
        if (absoluteCaptureIdx === i) {
            nodeCircle.classList.add('active-capture-node');
        }
        
        const isTarget = game.validDestinations.includes(i);
        const pulse = document.getElementById(`target-pulse-${i}`);
        if (isTarget) {
            nodeCircle.classList.add('valid-target');
            pulse.style.display = 'block';
        } else {
            pulse.style.display = 'none';
        }
        
        // Label inside node
        const label = document.getElementById(`node-label-${i}`);
        if (!hasA && !hasB) {
            label.textContent = displayNodeIndex;
            label.style.display = 'block';
        } else {
            label.style.display = 'none';
        }
        
        // 2. Update piece group
        const pieceG = document.getElementById(`piece-group-${i}`);
        if (hasA || hasB) {
            pieceG.style.display = 'block';
            pieceG.className.baseVal = `game-piece ${hasA ? 'green-player' : 'red-player'}`;
            if (game.selectedPiece === i) {
                pieceG.classList.add('selected');
            }
            
            const pieceText = document.getElementById(`piece-text-${i}`);
            pieceText.textContent = displayNodeIndex;
        } else {
            pieceG.style.display = 'none';
        }
    }
}

// Render logging panel
function renderHistory() {
    const log = document.getElementById('move-history-log');
    const returnBtn = document.getElementById('btn-return-current');
    log.innerHTML = '';
    
    if (game.moveLog.length === 0) {
        log.innerHTML = '<div class="history-empty">No moves made yet.</div>';
        returnBtn.style.display = 'none';
        return;
    }
    
    game.moveLog.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = `history-item ${item.includes('jump') ? 'capture' : ''}`;
        
        if (game.viewingHistoryIndex === index) {
            row.classList.add('active-history');
        }
        
        const idx = document.createElement('span');
        idx.className = 'history-index';
        idx.textContent = `${index + 1}.`;
        
        const txt = document.createElement('span');
        txt.className = 'history-move-text';
        txt.textContent = item;
        
        const time = document.createElement('span');
        time.className = 'history-time';
        time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        row.appendChild(idx);
        row.appendChild(txt);
        row.appendChild(time);
        
        // Make history item clickable to view that board state
        row.addEventListener('click', () => viewHistoryState(index));
        
        log.appendChild(row);
    });
    
    // Show/hide return button
    returnBtn.style.display = game.viewingHistoryIndex >= 0 ? 'block' : 'none';
    
    // Auto-scroll history log to bottom
    log.scrollTop = log.scrollHeight;
}

// View a past board state from a history snapshot
function viewHistoryState(index) {
    if (index < 0 || index >= game.moveSnapshots.length) return;
    
    const snapshot = game.moveSnapshots[index];
    game.viewingHistoryIndex = index;
    
    // Render the snapshot onto the board (read-only preview)
    renderBoardFromSnapshot(snapshot.me, snapshot.opp);
    renderHistory();
}

// Return to the live game state
function returnToCurrentGame() {
    game.viewingHistoryIndex = -1;
    renderBoard();
    renderHistory();
}

// Render a read-only board preview from explicit me/opp masks
function renderBoardFromSnapshot(meStr, oppStr) {
    const meMask = BigInt(meStr);
    const oppMask = BigInt(oppStr);
    
    for (let i = 0; i < 37; i++) {
        const hasA = (meMask & (1n << BigInt(i))) !== 0n;
        const hasB = (oppMask & (1n << BigInt(i))) !== 0n;
        
        // Node circles
        const nodeCircle = document.getElementById(`node-${i}`);
        nodeCircle.className.baseVal = "board-node-circle";
        
        const pulse = document.getElementById(`target-pulse-${i}`);
        pulse.style.display = 'none';
        
        // Label inside node
        const label = document.getElementById(`node-label-${i}`);
        if (!hasA && !hasB) {
            label.textContent = i;
            label.style.display = 'block';
        } else {
            label.style.display = 'none';
        }
        
        // Piece group
        const pieceG = document.getElementById(`piece-group-${i}`);
        if (hasA || hasB) {
            pieceG.style.display = 'block';
            pieceG.className.baseVal = `game-piece ${hasA ? 'green-player' : 'red-player'}`;
            
            const pieceText = document.getElementById(`piece-text-${i}`);
            pieceText.textContent = i;
        } else {
            pieceG.style.display = 'none';
        }
    }
}

// Click listener for pieces
function handlePieceClick(nodeId) {
    if (game.isAnimating) return;
    if (game.viewingHistoryIndex >= 0) return; // Block interaction during history replay
    if (!isHumanTurn() || game.gameStatus !== 'ONGOING') return;
    
    const activePlayerA = game.currentPlayer === 'A';
    const myMask = activePlayerA ? game.me : game.opp;
    
    // Node must contain own piece to be selected (nodeId is absolute)
    if (!getBit(myMask, nodeId)) {
        game.selectedPiece = null;
        game.validDestinations = [];
        updateUI();
        return;
    }
    
    const engineNodeId = activePlayerA ? nodeId : 36 - nodeId;
    
    // Multi-capture lock condition
    if (game.activeCaptureIdx !== -1 && game.activeCaptureIdx !== nodeId) {
        return; // Capturing piece is locked
    }
    
    if (game.selectedPiece === nodeId) {
        // Toggle selection
        game.selectedPiece = null;
        game.validDestinations = [];
    } else {
        game.selectedPiece = nodeId;
        // Filter possible moves that start at this nodeId
        game.validDestinations = game.possibleMoves
            .filter(m => m.from === engineNodeId)
            .map(m => activePlayerA ? m.to : 36 - m.to);
    }
    
    updateUI();
}

// Click listener for empty nodes (target destination move trigger)
async function handleNodeClick(nodeId) {
    if (game.isAnimating) return;
    if (game.viewingHistoryIndex >= 0) return; // Block interaction during history replay
    if (!isHumanTurn() || game.selectedPiece === null) return;
    
    if (game.validDestinations.includes(nodeId)) {
        const activePlayerA = game.currentPlayer === 'A';
        const engineFrom = activePlayerA ? game.selectedPiece : 36 - game.selectedPiece;
        const engineTo = activePlayerA ? nodeId : 36 - nodeId;
        
        const chosenMove = game.possibleMoves.find(m => m.from === engineFrom && m.to === engineTo);
        if (chosenMove) {
            await applyUserMove(chosenMove);
        }
    }
}

// Apply selected move for user player
async function applyUserMove(moveObj) {
    if (game.activeCaptureIdx === -1) {
        clearAITrail();
    }
    set_loading(true);
    game.pushHistory();
    
    try {
        const response = await fetch('/api/apply_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: {
                    me: game.me,
                    opp: game.opp,
                    activeCaptureIdx: game.activeCaptureIdx
                },
                moveId: moveObj.moveId,
                currentPlayer: game.currentPlayer
            })
        });
        
        const data = await response.json();
        
        // Log move description
        // Log move description with absolute coordinates
        const fromName = game.currentPlayer === 'A' ? moveObj.from : 36 - moveObj.from;
        const toName = game.currentPlayer === 'A' ? moveObj.to : 36 - moveObj.to;
        const capturedName = (moveObj.captured !== -1) ? (game.currentPlayer === 'A' ? moveObj.captured : 36 - moveObj.captured) : -1;
        
        let logTxt = `Player ${game.currentPlayer}: ${fromName} ──> ${toName}`;
        if (capturedName !== -1) {
            logTxt = `Player ${game.currentPlayer}: ${fromName} ──(jump over ${capturedName})──> ${toName}`;
        }
        game.moveLog.push(logTxt);
        
        // Draw permanent highlight trail for the move
        if (moveObj.moveId > 0) {
            if (capturedName !== -1) {
                const line1 = document.getElementById(`line-${fromName}-${capturedName}`) || document.getElementById(`line-${capturedName}-${fromName}`);
                const line2 = document.getElementById(`line-${capturedName}-${toName}`) || document.getElementById(`line-${toName}-${capturedName}`);
                if (line1) line1.classList.add('ai-trail');
                if (line2) line2.classList.add('ai-trail');
            } else {
                const line = document.getElementById(`line-${fromName}-${toName}`) || 
                             document.getElementById(`line-${toName}-${fromName}`);
                if (line) line.classList.add('ai-trail');
            }
        }
        
        // Wait for the visual slide animation to finish
        if (moveObj.moveId > 0) {
            await new Promise(resolve => {
                animate_move(fromName, toName, capturedName, resolve);
            });
        }
        
        // Update masks
        game.setMasks(data.nextState.me, data.nextState.opp, data.nextState.activeCaptureIdx);
        
        // Save board snapshot for history replay
        game.moveSnapshots.push({ me: game.me, opp: game.opp });
        
        // Settle turn transitions
        if (data.nextState.activeCaptureIdx === -1) {
            // Turn completed, swap player
            game.currentPlayer = game.currentPlayer === 'A' ? 'B' : 'A';
        } else {
            // locked in a capture chain, keep current turn
            game.selectedPiece = data.nextState.activeCaptureIdx;
        }

        // Render the new board layout immediately to avoid visual lag or glitches
        updateUI();
        
        // Check game status
        const statusResponse = await fetch('/api/game_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                me: game.me,
                opp: game.opp,
                activeCaptureIdx: game.activeCaptureIdx,
                currentPlayer: game.currentPlayer
            })
        });
        const statusData = await statusResponse.json();
        game.gameStatus = statusData.status;
        
        game.selectedPiece = null;
        game.validDestinations = [];
        updateUI();
        
        if (game.gameStatus === 'ONGOING') {
            await fetchPossibleMoves();
            
            // Trigger AI if applicable
            if (!isHumanTurn()) {
                setTimeout(triggerAIMove, 650);
            }
        }
    } catch (e) {
        console.error('Failed to apply move:', e);
        game.popHistory();
        updateUI();
    } finally {
        set_loading(false);
    }
}

// Trigger AI Calculation turn
async function triggerAIMove() {
    if (game.gameStatus !== 'ONGOING') return;
    
    if (game.activeCaptureIdx === -1) {
        clearAITrail();
    }
    
    set_loading(true, true);
    document.getElementById('metric-search-status').textContent = 'Searching...';
    
    try {
        const response = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: {
                    me: game.me,
                    opp: game.opp,
                    activeCaptureIdx: game.activeCaptureIdx
                },
                currentPlayer: game.currentPlayer,
                aiType: game.gameMode === 'pve_nn' ? 'nn' : 'minimax',
                depth: game.aiDepth
            })
        });
        
        const data = await response.json();
        
        if (data.moveId === -1) {
            // AI has no moves, player wins
            game.gameStatus = game.currentPlayer === 'A' ? 'OPP_WINS' : 'ME_WINS';
            updateUI();
            return;
        }
        
        // Log AI action with absolute coordinates
        const fromName = game.currentPlayer === 'A' ? data.move.from : 36 - data.move.from;
        const toName = game.currentPlayer === 'A' ? data.move.to : 36 - data.move.to;
        const capturedName = (data.move.captured !== -1) ? (game.currentPlayer === 'A' ? data.move.captured : 36 - data.move.captured) : -1;
        
        let logTxt = `AI (${game.currentPlayer}): ${fromName} ──> ${toName}`;
        if (data.moveId === 0) {
            logTxt = `AI (${game.currentPlayer}): Stop capture chain / Pass`;
        } else if (capturedName !== -1) {
            logTxt = `AI (${game.currentPlayer}): ${fromName} ──(jump over ${capturedName})──> ${toName}`;
        }
        game.moveLog.push(logTxt);
        
        // Draw temporary highlight line for AI move
        if (data.moveId > 0) {
            if (capturedName !== -1) {
                // Jump move - highlight both segments
                const line1 = document.getElementById(`line-${fromName}-${capturedName}`) || document.getElementById(`line-${capturedName}-${fromName}`);
                const line2 = document.getElementById(`line-${capturedName}-${toName}`) || document.getElementById(`line-${toName}-${capturedName}`);
                if (line1) line1.classList.add('ai-trail');
                if (line2) line2.classList.add('ai-trail');
            } else {
                // Adjacent move
                const line = document.getElementById(`line-${fromName}-${toName}`) || 
                             document.getElementById(`line-${toName}-${fromName}`);
                if (line) line.classList.add('ai-trail');
            }
        }
        
        // Wait for the visual slide animation to finish
        if (data.moveId > 0) {
            await new Promise(resolve => {
                animate_move(fromName, toName, capturedName, resolve);
            });
        }
        
        game.setMasks(data.nextState.me, data.nextState.opp, data.nextState.activeCaptureIdx);
        
        // Turn completed transition
        if (data.nextState.activeCaptureIdx === -1) {
            game.currentPlayer = game.currentPlayer === 'A' ? 'B' : 'A';
        }

        // Render the new board layout immediately to avoid visual lag or glitches
        updateUI();
        
        // Check game status
        const statusResponse = await fetch('/api/game_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                me: game.me,
                opp: game.opp,
                activeCaptureIdx: game.activeCaptureIdx,
                currentPlayer: game.currentPlayer
            })
        });
        const statusData = await statusResponse.json();
        game.gameStatus = statusData.status;
        
        updateUI();
        
        if (game.gameStatus === 'ONGOING') {
            await fetchPossibleMoves();
            
            // Loop turns if in AI Simulation (EvE)
            if (game.gameMode === 'eve') {
                setTimeout(triggerAIMove, 700);
            } else if (game.currentPlayer !== game.userRole) {
                // locked in further capture chain for AI
                setTimeout(triggerAIMove, 700);
            }
        }
    } catch (e) {
        console.error('Error during AI calculations:', e);
    } finally {
        set_loading(false);
        document.getElementById('metric-search-status').textContent = 'Idle';
    }
}

// User decides to terminate active capture chain early
async function handleEndChain() {
    if (game.isAnimating) return;
    if (!isHumanTurn() || game.activeCaptureIdx === -1) return;
    
    // END_CHAIN_MOVE is 0
    const endMove = { moveId: 0, from: -1, to: -1, captured: -1 };
    await applyUserMove(endMove);
}

// Undo action
function handleUndo() {
    if (game.isAnimating) return;
    if (!isHumanTurn() && game.gameMode !== 'eve') return; // Cannot undo during AI turn
    game.popHistory();
    updateUI();
    fetchPossibleMoves();
}

/**
 * @brief Animates the movement of a piece from fromNode to toNode.
 * @details Adds transition classes and calculates relative translations.
 */
function animate_move(fromNode, toNode, capturedNode, callback) {
    game.isAnimating = true;
    
    const fromCoord = getCoordinate(fromNode);
    const toCoord = getCoordinate(toNode);
    
    if (!fromCoord || !toCoord) {
        game.isAnimating = false;
        callback();
        return;
    }
    
    const dx = toCoord.x - fromCoord.x;
    const dy = toCoord.y - fromCoord.y;
    
    const pieceG = document.getElementById(`piece-group-${fromNode}`);
    if (pieceG) {
        // Bring the moving piece to the front/top by appending it to the end of its parent
        const parent = pieceG.parentNode;
        if (parent) {
            parent.appendChild(pieceG);
        }
        
        pieceG.classList.add('animating');
        pieceG.style.transform = `translate(${dx}px, ${dy}px)`;
    }
    
    let capturedPieceG = null;
    if (capturedNode !== -1) {
        capturedPieceG = document.getElementById(`piece-group-${capturedNode}`);
        if (capturedPieceG) {
            capturedPieceG.classList.add('captured-animating');
        }
    }
    
    setTimeout(() => {
        // Run callback first to update the game masks and re-render the board
        callback();
        
        // Clean up animation classes and transforms in the next frame after DOM updates are rendered
        requestAnimationFrame(() => {
            if (pieceG) {
                pieceG.style.transition = 'none';
                pieceG.style.transform = '';
                pieceG.classList.remove('animating');
                void pieceG.offsetHeight; // force reflow
                pieceG.style.transition = '';
            }
            if (capturedPieceG) {
                capturedPieceG.style.transition = 'none';
                capturedPieceG.style.opacity = '';
                capturedPieceG.style.transform = '';
                capturedPieceG.classList.remove('captured-animating');
                void capturedPieceG.offsetHeight; // force reflow
                capturedPieceG.style.transition = '';
            }
            game.isAnimating = false;
        });
    }, 400);
}

/**
 * @brief Non-blocking placeholder to satisfy the loading state requirement.
 * @details Prevents display of the blocking thinking overlay.
 */
function set_loading(isLoading, isAi = false) {
    // No-op to support seamless gameplay
}

// Show GameOver Modal dialog popup
function showGameOverModal() {
    const modal = document.getElementById('game-over-modal');
    const icon = document.getElementById('modal-result-icon');
    const title = document.getElementById('modal-result-title');
    const message = document.getElementById('modal-result-message');
    
    let winner = game.gameStatus === 'ME_WINS' ? 'A' : 'B';
    title.textContent = `Player ${winner} Wins!`;
    
    if (game.gameMode === 'pvp') {
        icon.textContent = '🏆';
        message.textContent = `Excellent match! Player ${winner} has captured all opponent pieces and dominates the board.`;
    } else if (game.gameMode.startsWith('pve')) {
        const humanWon = winner === game.userRole;
        if (humanWon) {
            icon.textContent = '🎉';
            message.textContent = "Spectacular! You successfully outsmarted the C++ AI Engine and won the game!";
        } else {
            icon.textContent = '🤖';
            message.textContent = "AI engine has captured all your pieces. Keep practicing to master the Sholo Guti board!";
        }
    } else { // EvE simulation
        icon.textContent = '📊';
        message.textContent = `AI simulation concluded. Player ${winner} AI won the match.`;
    }
    
    modal.style.display = 'flex';
}

// Clear AI move trail highlights
function clearAITrail() {
    document.querySelectorAll('.ai-trail').forEach(el => {
        el.classList.remove('ai-trail');
    });
}

// Bitboard manipulation utilities
function getBit(maskStr, bitIndex) {
    try {
        const mask = BigInt(maskStr);
        return (mask & (1n << BigInt(bitIndex))) !== 0n;
    } catch (e) {
        return false;
    }
}
