
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from multiprocessing import Pool
import io


def is_valid_board(board):
    return np.issubdtype(board.dtype, np.bool_)


def neighbors(i, j, rows, cols):
    return [
        (i, (j - 1) % cols), (i, (j + 1) % cols),
        ((i - 1) % rows, j), ((i + 1) % rows, j),
        ((i - 1) % rows, (j + 1) % cols), ((i + 1) % rows, (j + 1) % cols),
        ((i - 1) % rows, (j - 1) % cols), ((i + 1) % rows, (j - 1) % cols)
    ]


def gol_step(M):
    rows, cols = M.shape
    temp = M.copy()
    for i in range(rows):
        for j in range(cols):
            live_neighbors = sum(M[neighbor] for neighbor in neighbors(i, j, rows, cols))
            if M[i, j] and (live_neighbors < 2 or live_neighbors > 3):
                temp[i, j] = False
            elif not M[i, j] and live_neighbors == 3:
                temp[i, j] = True
    return temp


def center_pattern(board, pattern):
    start_row = (board.shape[0] - pattern.shape[0]) // 2
    start_col = (board.shape[1] - pattern.shape[1]) // 2
    board[start_row:start_row + pattern.shape[0], start_col:start_col + pattern.shape[1]] = pattern
    return board


def generate_frames_for_pattern(args):
    """Generate frames for a single pattern."""
    name, pattern, board_size, steps_per_pattern, pause_frames = args
    board = np.zeros(board_size, dtype=bool)
    board = center_pattern(board, pattern)
    frames = []
    current_board = board.copy()

    # Generate evolution frames
    for _ in range(steps_per_pattern):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.matshow(current_board, cmap='binary', interpolation='nearest')
        ax.set_title(name, color='white', fontsize=10, pad=10)
        ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=72, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        current_board = gol_step(current_board)

    # Add pause frames
    last_frame = frames[-1]
    for _ in range(pause_frames):
        frames.append(last_frame)

    return frames


def gif_gol_with_transitions_parallel(patterns, board_size=(70, 70), steps_per_pattern=50, pause_frames=20, transition_frames=10, gif_name="conways_game_of_life_black_white.gif"):
    # Prepare arguments for parallel processing
    args = [
        (name, pattern, board_size, steps_per_pattern, pause_frames)
        for name, pattern in patterns.items()
    ]

    # Use multiprocessing to process patterns in parallel
    with Pool() as pool:
        all_frames = pool.map(generate_frames_for_pattern, args)

    # Combine all frames
    combined_frames = []
    for idx, frames in enumerate(all_frames):
        combined_frames.extend(frames)
        if idx < len(all_frames) - 1:  # Add transition if not the last pattern
            last_frame = frames[-1]
            for alpha in np.linspace(1, 0, transition_frames):
                faded_frame = (last_frame * alpha).astype(np.uint8)
                combined_frames.append(faded_frame)

    # Save final GIF
    with imageio.get_writer(gif_name, mode='I', duration=0.05) as writer:
        for frame in combined_frames:
            writer.append_data(frame)

    print(f"Generated {gif_name}")

# Extensive Collection of Conway's Game of Life Patterns
patterns = {
    
    "Glider": np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ]),
    
    "Lightweight Spaceship (LWSS)": np.array([
        [False, True, False, False, True],
        [True, False, False, False, False],
        [True, False, False, False, True],
        [True, True, True, True, False]
    ]),
    
    "Heavyweight Spaceship (HWSS)": np.array([
        [False, False, True, True, False],
        [True, True, False, False, True],
        [False, True, False, False, True],
        [False, False, True, False, False],
        [True, True, True, True, False]
    ]),
    
    "Gosper Glider Gun": np.array([
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True],
        [True, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, True, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    ]),
    
    "Beacon": np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True]
    ]),
    
    "Pulsar": np.array([
        [False, False, True, True, True, False, False, False, True, True, True, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [True, False, False, False, False, True, False, True, False, False, False, False, True],
        [True, False, False, False, False, True, False, True, False, False, False, False, True],
        [True, False, False, False, False, True, False, True, False, False, False, False, True],
        [False, False, True, True, True, False, False, False, True, True, True, False, False]
    ]),
        
    "Blinker": np.array([
        [False, False, False],
        [True, True, True],
        [False, False, False]
    ]),
    
    "Toad": np.array([
        [False, False, False, False],
        [False, True, True, True],
        [True, True, True, False],
        [False, False, False, False]
    ]),
    
    "Pentadecathlon": np.array([
        [False, False, True, False, False, False, False, True, False, False],
        [True, True, False, True, True, True, True, False, True, True],
        [False, False, True, False, False, False, False, True, False, False]
    ]),
    
    "R-pentomino": np.array([
        [False, True, True],
        [True, True, False],
        [False, True, False]
    ]),
    
    "Diehard": np.array([
        [False, False, False, False, False, False, True, False],
        [True, True, False, False, False, False, False, False],
        [False, True, False, False, False, True, True, True]
    ]),

    "Acorn": np.array([
        [False, True, False, False, False, False, False],
        [False, False, False, True, False, False, False],
        [True, True, False, False, True, True, True]
    ]),

    "Small Exploder": np.array([
        [False, True, False],
        [True, True, True],
        [False, True, False],
        [False, True, False]
    ]),

    "Tumbler": np.array([
        [False, True, True, False, False, False, True, True, False],
        [False, True, True, False, False, False, True, True, False],
        [False, False, True, False, False, False, True, False, False],
        [True, False, True, False, False, False, True, False, True],
        [True, False, True, False, False, False, True, False, True],
        [True, False, True, False, False, False, True, False, True],
        [False, True, False, False, False, False, False, True, False]
    ])
}

# Create GIF with all patterns
gif_gol_with_transitions_parallel(
    patterns, 
    board_size=(80, 80),
    steps_per_pattern=50,
    pause_frames=20,
    transition_frames=10,
    gif_name="conways_game_of_life_all_patterns.gif"
)