import streamlit as st
import copy
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"

class Color(Enum):
    WHITE = "white"
    BLACK = "black"

class Piece:
    def __init__(self, piece_type: PieceType, color: Color):
        self.type = piece_type
        self.color = color
        self.has_moved = False
    
    def __str__(self):
        symbols = {
            (PieceType.KING, Color.WHITE): "♔",
            (PieceType.QUEEN, Color.WHITE): "♕",
            (PieceType.ROOK, Color.WHITE): "♖",
            (PieceType.BISHOP, Color.WHITE): "♗",
            (PieceType.KNIGHT, Color.WHITE): "♘",
            (PieceType.PAWN, Color.WHITE): "♙",
            (PieceType.KING, Color.BLACK): "♚",
            (PieceType.QUEEN, Color.BLACK): "♛",
            (PieceType.ROOK, Color.BLACK): "♜",
            (PieceType.BISHOP, Color.BLACK): "♝",
            (PieceType.KNIGHT, Color.BLACK): "♞",
            (PieceType.PAWN, Color.BLACK): "♟",
        }
        return symbols.get((self.type, self.color), "")

class Move:
    def __init__(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                 piece: Piece, captured_piece: Optional[Piece] = None,
                 is_castling: bool = False, is_en_passant: bool = False,
                 promotion_piece: Optional[PieceType] = None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece = piece
        self.captured_piece = captured_piece
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
        self.promotion_piece = promotion_piece

class ChessGame:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Color.WHITE
        self.move_history = []
        self.en_passant_target = None
        self.half_move_clock = 0
        self.full_move_number = 1
        self.game_over = False
        self.game_result = None
        self.kings = {Color.WHITE: (7, 4), Color.BLACK: (0, 4)}
        self.setup_board()
    
    def setup_board(self):
        # Set up pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.BLACK)
            self.board[6][col] = Piece(PieceType.PAWN, Color.WHITE)
        
        # Set up other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.BLACK)
            self.board[7][col] = Piece(piece_type, Color.WHITE)
    
    def get_piece_at(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8
    
    def get_legal_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        piece = self.get_piece_at(row, col)
        if not piece or piece.color != self.current_player:
            return []
        
        moves = []
        
        if piece.type == PieceType.PAWN:
            moves = self._get_pawn_moves(row, col)
        elif piece.type == PieceType.ROOK:
            moves = self._get_rook_moves(row, col)
        elif piece.type == PieceType.KNIGHT:
            moves = self._get_knight_moves(row, col)
        elif piece.type == PieceType.BISHOP:
            moves = self._get_bishop_moves(row, col)
        elif piece.type == PieceType.QUEEN:
            moves = self._get_queen_moves(row, col)
        elif piece.type == PieceType.KING:
            moves = self._get_king_moves(row, col)
        
        # Filter out moves that would leave the king in check
        legal_moves = []
        for move in moves:
            if self._is_legal_move(row, col, move[0], move[1]):
                legal_moves.append(move)
        
        return legal_moves
    
    def _get_pawn_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        piece = self.get_piece_at(row, col)
        direction = -1 if piece.color == Color.WHITE else 1
        start_row = 6 if piece.color == Color.WHITE else 1
        
        # Forward move
        new_row = row + direction
        if self.is_valid_position(new_row, col) and not self.get_piece_at(new_row, col):
            moves.append((new_row, col))
            
            # Two squares forward from starting position
            if row == start_row and not self.get_piece_at(new_row + direction, col):
                moves.append((new_row + direction, col))
        
        # Captures
        for dc in [-1, 1]:
            new_col = col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.get_piece_at(new_row, new_col)
                if target and target.color != piece.color:
                    moves.append((new_row, new_col))
                # En passant
                elif self.en_passant_target == (new_row, new_col):
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_rook_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target = self.get_piece_at(new_row, new_col)
                if target:
                    if target.color != self.get_piece_at(row, col).color:
                        moves.append((new_row, new_col))
                    break
                else:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_knight_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.get_piece_at(new_row, new_col)
                if not target or target.color != self.get_piece_at(row, col).color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target = self.get_piece_at(new_row, new_col)
                if target:
                    if target.color != self.get_piece_at(row, col).color:
                        moves.append((new_row, new_col))
                    break
                else:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_queen_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        return self._get_rook_moves(row, col) + self._get_bishop_moves(row, col)
    
    def _get_king_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.get_piece_at(new_row, new_col)
                if not target or target.color != self.get_piece_at(row, col).color:
                    moves.append((new_row, new_col))
        
        # Castling
        if not self.get_piece_at(row, col).has_moved and not self.is_in_check(self.current_player):
            # Kingside castling
            if (self.get_piece_at(row, 7) and 
                not self.get_piece_at(row, 7).has_moved and
                not self.get_piece_at(row, 5) and not self.get_piece_at(row, 6)):
                if not self._would_be_in_check_after_move(row, col, row, 5):
                    moves.append((row, 6))
            
            # Queenside castling
            if (self.get_piece_at(row, 0) and 
                not self.get_piece_at(row, 0).has_moved and
                not self.get_piece_at(row, 1) and not self.get_piece_at(row, 2) and not self.get_piece_at(row, 3)):
                if not self._would_be_in_check_after_move(row, col, row, 3):
                    moves.append((row, 2))
        
        return moves
    
    def _is_legal_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        return not self._would_be_in_check_after_move(from_row, from_col, to_row, to_col)
    
    def _would_be_in_check_after_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        # Make a temporary move
        original_piece = self.get_piece_at(to_row, to_col)
        piece = self.get_piece_at(from_row, from_col)
        
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Update king position if king moved
        if piece.type == PieceType.KING:
            self.kings[piece.color] = (to_row, to_col)
        
        in_check = self.is_in_check(self.current_player)
        
        # Undo the temporary move
        self.board[from_row][from_col] = piece
        self.board[to_row][to_col] = original_piece
        
        if piece.type == PieceType.KING:
            self.kings[piece.color] = (from_row, from_col)
        
        return in_check
    
    def is_in_check(self, color: Color) -> bool:
        king_pos = self.kings[color]
        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        
        # Check if any opponent piece can attack the king
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if piece and piece.color == opponent_color:
                    if self._can_attack(row, col, king_pos[0], king_pos[1]):
                        return True
        return False
    
    def _can_attack(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        piece = self.get_piece_at(from_row, from_col)
        if not piece:
            return False
        
        if piece.type == PieceType.PAWN:
            direction = -1 if piece.color == Color.WHITE else 1
            if from_row + direction == to_row and abs(from_col - to_col) == 1:
                return True
        elif piece.type == PieceType.ROOK:
            if from_row == to_row or from_col == to_col:
                return self._is_path_clear(from_row, from_col, to_row, to_col)
        elif piece.type == PieceType.KNIGHT:
            dr, dc = abs(from_row - to_row), abs(from_col - to_col)
            if (dr == 2 and dc == 1) or (dr == 1 and dc == 2):
                return True
        elif piece.type == PieceType.BISHOP:
            if abs(from_row - to_row) == abs(from_col - to_col):
                return self._is_path_clear(from_row, from_col, to_row, to_col)
        elif piece.type == PieceType.QUEEN:
            if (from_row == to_row or from_col == to_col or 
                abs(from_row - to_row) == abs(from_col - to_col)):
                return self._is_path_clear(from_row, from_col, to_row, to_col)
        elif piece.type == PieceType.KING:
            if abs(from_row - to_row) <= 1 and abs(from_col - to_col) <= 1:
                return True
        
        return False
    
    def _is_path_clear(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        dr = 0 if from_row == to_row else (1 if to_row > from_row else -1)
        dc = 0 if from_col == to_col else (1 if to_col > from_col else -1)
        
        row, col = from_row + dr, from_col + dc
        while row != to_row or col != to_col:
            if self.get_piece_at(row, col):
                return False
            row += dr
            col += dc
        
        return True
    
    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int, promotion_piece: Optional[PieceType] = None) -> bool:
        piece = self.get_piece_at(from_row, from_col)
        if not piece or piece.color != self.current_player:
            return False
        
        legal_moves = self.get_legal_moves(from_row, from_col)
        if (to_row, to_col) not in legal_moves:
            return False
        
        # Handle special moves
        captured_piece = self.get_piece_at(to_row, to_col)
        is_castling = False
        is_en_passant = False
        
        # Check for castling
        if piece.type == PieceType.KING and abs(to_col - from_col) == 2:
            is_castling = True
            # Move the rook
            if to_col == 6:  # Kingside
                rook = self.get_piece_at(from_row, 7)
                self.board[from_row][5] = rook
                self.board[from_row][7] = None
                rook.has_moved = True
            else:  # Queenside
                rook = self.get_piece_at(from_row, 0)
                self.board[from_row][3] = rook
                self.board[from_row][0] = None
                rook.has_moved = True
        
        # Check for en passant
        if (piece.type == PieceType.PAWN and 
            self.en_passant_target == (to_row, to_col)):
            is_en_passant = True
            # Remove the captured pawn
            capture_row = to_row + (1 if piece.color == Color.WHITE else -1)
            captured_piece = self.get_piece_at(capture_row, to_col)
            self.board[capture_row][to_col] = None
        
        # Make the move
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        piece.has_moved = True
        
        # Update king position
        if piece.type == PieceType.KING:
            self.kings[piece.color] = (to_row, to_col)
        
        # Handle pawn promotion
        if (piece.type == PieceType.PAWN and 
            ((piece.color == Color.WHITE and to_row == 0) or 
             (piece.color == Color.BLACK and to_row == 7))):
            if promotion_piece:
                self.board[to_row][to_col] = Piece(promotion_piece, piece.color)
            else:
                self.board[to_row][to_col] = Piece(PieceType.QUEEN, piece.color)
        
        # Update en passant target
        if (piece.type == PieceType.PAWN and abs(to_row - from_row) == 2):
            self.en_passant_target = ((from_row + to_row) // 2, to_col)
        else:
            self.en_passant_target = None
        
        # Record the move
        move = Move(
            (from_row, from_col), (to_row, to_col), piece, captured_piece,
            is_castling, is_en_passant, promotion_piece
        )
        self.move_history.append(move)
        
        # Update move counters
        if piece.type == PieceType.PAWN or captured_piece:
            self.half_move_clock = 0
        else:
            self.half_move_clock += 1
        
        if self.current_player == Color.BLACK:
            self.full_move_number += 1
        
        # Switch turns
        self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        
        # Check for game end
        self._check_game_end()
        
        return True
    
    def _check_game_end(self):
        # Check if current player has any legal moves
        has_legal_moves = False
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if piece and piece.color == self.current_player:
                    if self.get_legal_moves(row, col):
                        has_legal_moves = True
                        break
            if has_legal_moves:
                break
        
        if not has_legal_moves:
            if self.is_in_check(self.current_player):
                # Checkmate
                winner = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
                self.game_result = f"Checkmate! {winner.value.title()} wins!"
            else:
                # Stalemate
                self.game_result = "Stalemate! It's a draw!"
            self.game_over = True
        
        # Check for draw by 50-move rule
        if self.half_move_clock >= 50:
            self.game_result = "Draw by 50-move rule!"
            self.game_over = True
        
        # Check for insufficient material
        if self._is_insufficient_material():
            self.game_result = "Draw by insufficient material!"
            self.game_over = True
    
    def _is_insufficient_material(self) -> bool:
        pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if piece and piece.type != PieceType.KING:
                    pieces.append(piece.type)
        
        if not pieces:
            return True
        
        if len(pieces) == 1 and pieces[0] in [PieceType.BISHOP, PieceType.KNIGHT]:
            return True
        
        return False
    
    def reset_game(self):
        self.__init__()

# Streamlit UI
def main():
    st.set_page_config(page_title="Chess Game", layout="wide")
    st.title("♟️ Chess Game")
    
    # Initialize game state
    if 'game' not in st.session_state:
        st.session_state.game = ChessGame()
    
    if 'selected_square' not in st.session_state:
        st.session_state.selected_square = None
    
    if 'legal_moves' not in st.session_state:
        st.session_state.legal_moves = []
    
    if 'promotion_needed' not in st.session_state:
        st.session_state.promotion_needed = False
    
    if 'promotion_move' not in st.session_state:
        st.session_state.promotion_move = None
    
    game = st.session_state.game
    
    # Game controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("New Game"):
            st.session_state.game = ChessGame()
            st.session_state.selected_square = None
            st.session_state.legal_moves = []
            st.session_state.promotion_needed = False
            st.session_state.promotion_move = None
            st.rerun()
    
    with col2:
        if game.game_over:
            st.error(game.game_result)
        else:
            current_player = game.current_player.value.title()
            if game.is_in_check(game.current_player):
                st.warning(f"{current_player} to move (In Check!)")
            else:
                st.info(f"{current_player} to move")
    
    # Handle pawn promotion
    if st.session_state.promotion_needed:
        st.subheader("Pawn Promotion")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("♕ Queen"):
                handle_promotion(PieceType.QUEEN)
        
        with col2:
            if st.button("♖ Rook"):
                handle_promotion(PieceType.ROOK)
        
        with col3:
            if st.button("♗ Bishop"):
                handle_promotion(PieceType.BISHOP)
        
        with col4:
            if st.button("♘ Knight"):
                handle_promotion(PieceType.KNIGHT)
    
    # Chess board
    st.subheader("Chess Board")
    
    # Create the board
    board_container = st.container()
    
    with board_container:
        for row in range(8):
            cols = st.columns(8)
            for col in range(8):
                with cols[col]:
                    square_color = "lightgray" if (row + col) % 2 == 0 else "darkgray"
                    piece = game.get_piece_at(row, col)
                    
                    # Highlight selected square and legal moves
                    if st.session_state.selected_square == (row, col):
                        square_color = "yellow"
                    elif (row, col) in st.session_state.legal_moves:
                        square_color = "lightgreen"
                    
                    piece_text = str(piece) if piece else ""
                    
                    if st.button(
                        piece_text,
                        key=f"square_{row}_{col}",
                        help=f"Row {row+1}, Col {chr(ord('a')+col)}",
                        disabled=game.game_over
                    ):
                        handle_square_click(row, col)
    
    # Move history
    if game.move_history:
        st.subheader("Move History")
        moves_text = []
        for i, move in enumerate(game.move_history):
            move_num = (i // 2) + 1
            color = "White" if i % 2 == 0 else "Black"
            from_square = f"{chr(ord('a')+move.from_pos[1])}{8-move.from_pos[0]}"
            to_square = f"{chr(ord('a')+move.to_pos[1])}{8-move.to_pos[0]}"
            
            if move.is_castling:
                move_text = "O-O" if move.to_pos[1] == 6 else "O-O-O"
            else:
                piece_symbol = ""
                if move.piece.type != PieceType.PAWN:
                    piece_symbol = move.piece.type.value[0].upper()
                
                capture = "x" if move.captured_piece or move.is_en_passant else ""
                move_text = f"{piece_symbol}{capture}{to_square}"
                
                if move.promotion_piece:
                    move_text += f"={move.promotion_piece.value[0].upper()}"
            
            if i % 2 == 0:
                moves_text.append(f"{move_num}. {move_text}")
            else:
                moves_text[-1] += f" {move_text}"
        
        st.text(" ".join(moves_text))

def handle_square_click(row: int, col: int):
    game = st.session_state.game
    
    if st.session_state.selected_square is None:
        # Select a piece
        piece = game.get_piece_at(row, col)
        if piece and piece.color == game.current_player:
            st.session_state.selected_square = (row, col)
            st.session_state.legal_moves = game.get_legal_moves(row, col)
    else:
        # Try to move
        from_row, from_col = st.session_state.selected_square
        
        if (row, col) == st.session_state.selected_square:
            # Deselect
            st.session_state.selected_square = None
            st.session_state.legal_moves = []
        elif (row, col) in st.session_state.legal_moves:
            # Check if this is a pawn promotion
            piece = game.get_piece_at(from_row, from_col)
            if (piece.type == PieceType.PAWN and 
                ((piece.color == Color.WHITE and row == 0) or 
                 (piece.color == Color.BLACK and row == 7))):
                # Need promotion
                st.session_state.promotion_needed = True
                st.session_state.promotion_move = (from_row, from_col, row, col)
            else:
                # Make the move
                game.make_move(from_row, from_col, row, col)
                st.session_state.selected_square = None
                st.session_state.legal_moves = []
        else:
            # Select a different piece
            piece = game.get_piece_at(row, col)
            if piece and piece.color == game.current_player:
                st.session_state.selected_square = (row, col)
                st.session_state.legal_moves = game.get_legal_moves(row, col)
            else:
                st.session_state.selected_square = None
                st.session_state.legal_moves = []
    
    st.rerun()

def handle_promotion(promotion_piece: PieceType):
    game = st.session_state.game
    from_row, from_col, to_row, to_col = st.session_state.promotion_move
    
    game.make_move(from_row, from_col, to_row, to_col, promotion_piece)
    
    st.session_state.selected_square = None
    st.session_state.legal_moves = []
    st.session_state.promotion_needed = False
    st.session_state.promotion_move = None
    
    st.rerun()

if __name__ == "__main__":
    main()
