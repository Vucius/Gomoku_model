import numpy as np

def apply_d4_symmetry(board, policy, sym_idx):
    """
    Applies one of the 8 D4 symmetry transformations to the input board and policy.
    Args:
        board: numpy array of shape (C, H, W)
        policy: numpy array of shape (H, W)
        sym_idx: integer in [0, 7] representing the symmetry index
    Returns:
        transformed_board: numpy array of shape (C, H, W)
        transformed_policy: numpy array of shape (H, W)
    """
    # sym_idx is split into:
    # - rotation: sym_idx % 4 (0, 90, 180, 270 degrees)
    # - flip: sym_idx // 4 (0: no flip, 1: horizontal flip)
    
    rot_k = sym_idx % 4
    flip_h = sym_idx // 4

    # Apply to board: channels are preserved, transform H and W
    x = board.copy()
    if flip_h:
        # Flip along horizontal axis (left-right flip, which flips the column index)
        # For a 3D array (C, H, W), flipping columns is axis 2
        x = np.flip(x, axis=2)
    if rot_k > 0:
        # Rotate in the plane of the last two dimensions (H, W), which are axes 1 and 2
        x = np.rot90(x, k=rot_k, axes=(1, 2))

    # Apply to policy (H, W)
    y = policy.copy()
    if flip_h:
        # Flip columns (axis 1)
        y = np.flip(y, axis=1)
    if rot_k > 0:
        # Rotate axes (0, 1)
        y = np.rot90(y, k=rot_k, axes=(0, 1))

    return x.copy(), y.copy()

def apply_player_swap(board, value_target):
    """
    Swaps the perspective of the players (swaps active player plane and opponent plane).
    Args:
        board: numpy array of shape (C, H, W) where board[0] is active player, board[1] is opponent
        value_target: float in [-1.0, 1.0] representing current player's win value
    Returns:
        swapped_board: board with active and opponent planes swapped
        swapped_value: -value_target
    """
    swapped_board = board.copy()
    # Swap plane 0 and plane 1
    swapped_board[0], swapped_board[1] = board[1].copy(), board[0].copy()
    
    return swapped_board, -value_target
