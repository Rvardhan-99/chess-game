import streamlit as st
import random

# Initialize the chessboard
def init_board():
    return [
        ['♜', '♞', '♝', '♛', '♚', '♝', '♞', '♜'],  # Black pieces
        ['♟', '♟', '♟', '♟', '♟', '♟', '♟', '♟'],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['♙', '♙', '♙', '♙', '♙', '♙', '♙', '♙'],  # White pieces
        ['♖', '♘', '♗', '♕', '♔', '♗', '♘', '♖']
    ]

# Validate piece movement (simplified rules)
def is_valid_move(board, start, end, player):
    start_row, start_col = start
    end_row, end_col = end
    piece = board[start_row][start_col]
    
    # Check if it's the player's piece
    if player == 'White' and piece not in ['♙', '♖', '♘', '♗', '♕', '♔']:
        return False, "Not your piece!"
    if player == 'Black' and piece not in ['♟', '♜', '♞', '♝', '♛', '♚']:
        return False, "Not your piece!"
    
    # Check if destination is valid
    if board[end_row][end_col] != ' ' and (
        (player == 'White' and board[end_row][end_col] in ['♙', '♖', '♘', '♗', '♕', '♔']) or
        (player == 'Black' and board[end_row][end_col] in ['♟', '♜', '♞', '♝', '♛', '♚'])
    ):
        return False, "Cannot capture your own piece!"
    
    # Simplified movement rules
    row_diff = end_row - start_row
    col_diff = end_col - start_col
    
    if piece in ['♙', '♟']:  # Pawn
        direction = -1 if player == 'White' else 1
        # Move forward
        if col_diff == 0 and row_diff == direction and board[end_row][end_col] == ' ':
            return True, ""
        # Capture diagonally
        if abs(col_diff) == 1 and row_diff == direction and board[end_row][end_col] != ' ':
            return True, ""
        return False, "Invalid pawn move!"
    
    if piece in ['♖', '♜']:  # Rook
        if row_diff == 0 or col_diff == 0:
            return check_path_clear(board, start, end), ""
        return False, "Invalid rook move!"
    
    if piece in ['♘', '♞']:  # Knight
        if (abs(row_diff), abs(col_diff)) in [(2, 1), (1, 2)]:
            return True, ""
        return False, "Invalid knight move!"
    
    if piece in ['♗', '♝']:  # Bishop
        if abs(row_diff) == abs(col_diff):
            return check_path_clear(board, start, end), ""
        return False, "Invalid bishop move!"
    
    if piece in ['♕', '♛']:  # Queen
        if row_diff == 0 or col_diff == 0 or abs(row_diff) == abs(col_diff):
            return check_path_clear(board, start, end), ""
        return False, "Invalid queen move!"
    
    if piece in ['♔', '♚']:  # King
        if max(abs(row_diff), abs(col_diff)) == 1:
            return True, ""
        return False, "Invalid king move!"
    
    return False, "Unknown piece!"

# Check if path is clear for rook, bishop, queen
def check_path_clear(board, start, end):
    start_row, start_col = start
    end_row, end_col = end
    row_step = 0 if end_row == start_row else (1 if end_row > start_row else -1)
    col_step = 0 if end_col == start_col else (1 if end_col > start_col else -1)
    steps = max(abs(end_row - start_row), abs(end_col - start_col))
    
    for i in range(1, steps):
        row = start_row + i * row_step
        col = start_col + i * col_step
        if board[row][col] != ' ':
            return False
    return True

# Check for checkmate or stalemate (simplified)
def check_game_over(board, player):
    # Placeholder: Check for checkmate or stalemate
    # For simplicity, only check if king is captured (not realistic, but simplifies demo)
    for i in range(8):
        for j in range(8):
            if (player == 'White' and board[i][j] == '♔') or (player == 'Black' and board[i][j] == '♚'):
                return False, ""
    return True, f"Game over! {player} loses (king captured)."

# Make a move
def make_move(start, end):
    if st.session_state.game_over:
        return
    valid, message = is_valid_move(st.session_state.board, start, end, st.session_state.current_player)
    if not valid:
        st.session_state.status = message
        return
    # Move piece
    board = st.session_state.board
    board[end[0]][end[1]] = board[start[0]][start[1]]
    board[start[0]][start[1]] = ' '
    # Check game over
    st.session_state.game_over, st.session_state.status = check_game_over(board, st.session_state.current_player)
    if not st.session_state.game_over:
        st.session_state.current_player = 'Black' if st.session_state.current_player == 'White' else 'White'
        st.session_state.status = f"Player {st.session_state.current_player}'s turn"
    st.session_state.selected = None

# Reset the game
def reset_game():
    st.session_state.board = init_board()
    st.session_state.current_player = 'White'
    st.session_state.game_over = False
    st.session_state.status = "Player White's turn"
    st.session_state.selected = None

# Main Streamlit app
def main():
    st.title("Chess Game")
    
    # Sidebar with developer credit
    st.sidebar.title("About")
    st.sidebar.write("Developed by Raja")
    
    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = init_board()
        st.session_state.current_player = 'White'
        st.session_state.game_over = False
        st.session_state.status = "Player White's turn"
        st.session_state.selected = None
    
    # Display game status
    st.write(st.session_state.status)
    
    # Create 8x8 grid of buttons
    for i in range(8):
        cols = st.columns(8)
        for j in range(8):
            with cols[j]:
                button_label = st.session_state.board[i][j] if st.session_state.board[i][j] != ' ' else ' '
                if st.button(button_label, key=f"cell_{i}_{j}", disabled=st.session_state.game_over):
                    if st.session_state.selected is None:
                        if st.session_state.board[i][j] != ' ':
                            st.session_state.selected = (i, j)
                            st.session_state.status = f"Selected {button_label}. Click destination."
                    else:
                        make_move(st.session_state.selected, (i, j))
                        st.rerun()
    
    # Reset button
    if st.button("Reset Game"):
        reset_game()
        st.rerun()

if __name__ == "__main__":
    main()
