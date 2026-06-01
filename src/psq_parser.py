import os
import re

def check_five_in_a_row(board, board_size=15):
    """
    Checks if there is a 5-in-a-row on the board.
    Returns 1 if player 1 (Black) won, -1 if player -1 (White) won, or 0 if no win detected.
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for r in range(board_size):
        for c in range(board_size):
            player = board[r][c]
            if player == 0:
                continue
            for dr, dc in directions:
                # Check 5 stones in this direction
                win = True
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr][nc] != player:
                        win = False
                        break
                if win:
                    return int(player)
    return 0

def detect_threats(board, player, board_size=15):
    """
    Analyzes the board for threats for the given player.
    Returns a grid of shape (board_size, board_size) containing threat levels (0 to 5) for each empty cell:
    0: None
    1: Live Three (活三) - placing a stone creates an open-ended 4-in-a-row.
    2: Rush Four (冲四) - placing a stone creates a closed-ended 5-in-a-row (winning move but blocked at one end).
    3: Live Four (活四) - placing a stone creates a 5-in-a-row or double-open 4.
    Actually, let's keep it simple and robust by checking the length of consecutive stones
    that would be formed by placing a stone at each empty cell (x, y).
    """
    threat_grid = [[0 for _ in range(board_size)] for _ in range(board_size)]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] != 0:
                continue  # Threat is evaluated on empty squares
            
            # Temporary place stone
            board[r][c] = player
            
            # Check maximum consecutive line length passing through (r,c)
            max_len = 0
            for dr, dc in directions:
                length = 1
                # Forward
                i = 1
                while True:
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == player:
                        length += 1
                        i += 1
                    else:
                        break
                # Backward
                i = 1
                while True:
                    nr, nc = r - dr * i, c - dc * i
                    if 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == player:
                        length += 1
                        i += 1
                    else:
                        break
                max_len = max(max_len, length)
            
            # Revert
            board[r][c] = 0

            # Map length of consecutive stones to threat classes
            # (Note: we use a simplified robust heuristic)
            if max_len >= 5:
                threat_grid[r][c] = 3  # Live Four / Win
            elif max_len == 4:
                threat_grid[r][c] = 2  # Rush Four / Four-in-a-row
            elif max_len == 3:
                threat_grid[r][c] = 1  # Live Three / Three-in-a-row
            else:
                threat_grid[r][c] = 0

    return threat_grid

def parse_psq_file(filepath, board_size=15):
    """
    Parses a single Gomocup .psq file.
    Returns:
        moves: List of (r, c) tuples, 0-indexed.
        winner: 1 for Black (first player), -1 for White (second player), 0 for draw.
        states_seq: List of dicts representing the game steps.
    """
    moves = []
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Match coordinates like "8,7,0" or "8,7,12345"
    coord_pattern = re.compile(r'^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Stop on indicator or tournament metadata
        if line.startswith("-1") or ".zip" in line.lower():
            break

        match = coord_pattern.match(line)
        if match:
            # PSQ is 1-indexed (1 to 15), so convert to 0-indexed (0 to 14)
            x = int(match.group(1)) - 1
            y = int(match.group(2)) - 1
            
            # Ensure coordinates are within valid range
            if 0 <= x < board_size and 0 <= y < board_size:
                moves.append((y, x))  # Standard row=y, col=x

    # Reconstruct board states and determine winner
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    states_seq = []
    current_player = 1  # Black starts

    for step, (r, c) in enumerate(moves):
        # Create a snapshot of current board state before the move
        board_snapshot = [row[:] for row in board]
        
        # Save game step
        states_seq.append({
            "board": board_snapshot,
            "player": current_player,
            "move": (r, c),
            "step": step
        })
        
        # Play the move
        board[r][c] = current_player
        
        # Switch player
        current_player = -current_player

    # Determine winner from final board
    winner = check_five_in_a_row(board, board_size)
    
    # If no 5-in-a-row, check if the game ended abruptly and default to last mover
    if winner == 0 and len(moves) > 0:
        # Check if the last mover has a win (sometimes 5-in-a-row check misses subtle rules,
        # but 5-in-a-row is standard. Otherwise we label it a draw)
        winner = 0

    return {
        "moves": moves,
        "winner": winner,
        "states_seq": states_seq
    }
