import random
import math
import copy
import pygame
import sys
import ctypes
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ————————————————————————
# Constants
# ————————————————————————
ROWS, COLUMNS = 6, 7
EMPTY, P1, P2 = 0, 1, 2
WINDOW_LENGTH = 4

SQUARESIZE = 100
RADIUS     = int(SQUARESIZE/2 - 5)
FPS        = 60

BLUE   = (0,   0,   255)
BLACK  = (0,   0,   0)
RED    = (255, 0,   0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)
GRAY   = (200, 200, 200)
HOVER  = (170, 170, 170)

# ————————————————————————
# Metrics storage
# ————————————————————————
metrics = {
    'nodes':     {P1: [], P2: []},
    'times':     {P1: [], P2: []},
    'heuristic': {P1: [], P2: []},
    'delta':     {P1: [], P2: []},
}

minimax_counter   = 0
alphabeta_counter = 0
mcts_counter      = 0

# ————————————————————————
# Game logic
# ————————————————————————
class ConnectFour:
    def __init__(self):
        self.board = [[EMPTY]*COLUMNS for _ in range(ROWS)]

    def drop_piece(self, col, piece):
        for r in range(ROWS-1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = piece
                return True
        return False

    def valid_moves(self):
        return [c for c in range(COLUMNS) if self.board[0][c] == EMPTY]

    def is_full(self):
        return all(self.board[0][c] != EMPTY for c in range(COLUMNS))

    def winning_move(self, piece):
        # horizontal
        for r in range(ROWS):
            for c in range(COLUMNS-3):
                if all(self.board[r][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True
        # vertical
        for c in range(COLUMNS):
            for r in range(ROWS-3):
                if all(self.board[r+i][c] == piece for i in range(WINDOW_LENGTH)):
                    return True
        # diagonals
        for r in range(ROWS-3):
            for c in range(COLUMNS-3):
                if all(self.board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True
                if all(self.board[r+3-i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True
        return False

    def score_window(self, window, piece):
        score = 0
        opp = P1 if piece == P2 else P2
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2
        if window.count(opp) == 3 and window.count(EMPTY) == 1:
            score -= 4
        return score

    def evaluate(self, piece):
        score = 0
        # center column
        center_col = [self.board[r][COLUMNS//2] for r in range(ROWS)]
        score += center_col.count(piece)*3
        # horizontals
        for r in range(ROWS):
            for c in range(COLUMNS-3):
                score += self.score_window(self.board[r][c:c+WINDOW_LENGTH], piece)
        # verticals
        for c in range(COLUMNS):
            for r in range(ROWS-3):
                col_arr = [self.board[r+i][c] for i in range(WINDOW_LENGTH)]
                score += self.score_window(col_arr, piece)
        # diagonals
        for r in range(ROWS-3):
            for c in range(COLUMNS-3):
                diag1 = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                diag2 = [self.board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.score_window(diag1, piece)
                score += self.score_window(diag2, piece)
        return score

    def copy(self):
        new = ConnectFour()
        new.board = copy.deepcopy(self.board)
        return new

# ————————————————————————
# Minimax with node counting
# ————————————————————————
def minimax(board, depth, maximizing):
    global minimax_counter
    minimax_counter += 1
    valid = board.valid_moves()
    term = board.winning_move(P1) or board.winning_move(P2) or board.is_full()
    if depth == 0 or term:
        if term:
            if board.winning_move(P2): return None, 1e9
            if board.winning_move(P1): return None, -1e9
            return None, 0
        return None, board.evaluate(P2)
    if maximizing:
        value, choice = -math.inf, random.choice(valid)
        for c in valid:
            b2 = board.copy()
            b2.drop_piece(c, P2)
            sc = minimax(b2, depth-1, False)[1]
            if sc > value:
                value, choice = sc, c
        return choice, value
    else:
        value, choice = math.inf, random.choice(valid)
        for c in valid:
            b2 = board.copy()
            b2.drop_piece(c, P1)
            sc = minimax(b2, depth-1, True)[1]
            if sc < value:
                value, choice = sc, c
        return choice, value

# ————————————————————————
# Alpha-Beta with node counting
# ————————————————————————
def alphabeta(board, depth, alpha, beta, maximizing):
    global alphabeta_counter
    alphabeta_counter += 1
    valid = board.valid_moves()
    term = board.winning_move(P1) or board.winning_move(P2) or board.is_full()
    if depth == 0 or term:
        if term:
            if board.winning_move(P2): return None, 1e9
            if board.winning_move(P1): return None, -1e9
            return None, 0
        return None, board.evaluate(P2)
    if maximizing:
        value, choice = -math.inf, random.choice(valid)
        for c in valid:
            b2 = board.copy()
            b2.drop_piece(c, P2)
            sc = alphabeta(b2, depth-1, alpha, beta, False)[1]
            if sc > value:
                value, choice = sc, c
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return choice, value
    else:
        value, choice = math.inf, random.choice(valid)
        for c in valid:
            b2 = board.copy()
            b2.drop_piece(c, P1)
            sc = alphabeta(b2, depth-1, alpha, beta, True)[1]
            if sc < value:
                value, choice = sc, c
            beta = min(beta, value)
            if alpha >= beta:
                break
        return choice, value

# ————————————————————————
# MCTS with simulation counting
# ————————————————————————
class MCTSNode:
    def __init__(self, board, piece, parent=None):
        self.board, self.piece, self.parent = board, piece, parent
        self.children = {}
        self.visits = 0
        self.wins   = 0

    def fully_expanded(self):
        return len(self.children) == len(self.board.valid_moves())

    def best_child(self, c=1.4):
        best, best_uct = None, -1
        for mv, node in self.children.items():
            uct = node.wins/node.visits + c*math.sqrt(math.log(self.visits)/node.visits)
            if uct > best_uct:
                best_uct, best = uct, node
        return best

    def expand(self):
        tried = set(self.children.keys())
        moves = [m for m in self.board.valid_moves() if m not in tried]
        m = random.choice(moves)
        b2 = self.board.copy()
        b2.drop_piece(m, self.piece)
        nxt = P1 if self.piece == P2 else P2
        child = MCTSNode(b2, nxt, self)
        self.children[m] = child
        return child

    def rollout(self):
        b2, p = self.board.copy(), self.piece
        while True:
            mv = b2.valid_moves()
            if not mv:
                return 0
            m = random.choice(mv)
            b2.drop_piece(m, p)
            if b2.winning_move(p):
                return 1 if p == P2 else -1
            p = P1 if p == P2 else P2

    def backprop(self, res):
        self.visits += 1
        self.wins   += (res == 1) + 0.5*(res == 0)
        if self.parent:
            self.parent.backprop(res)

def mcts(root_board, sims):
    global mcts_counter
    mcts_counter = 0
    root = MCTSNode(root_board.copy(), P2)
    for _ in range(sims):
        mcts_counter += 1
        node = root
        while node.fully_expanded() and node.children:
            node = node.best_child()
        if not node.board.winning_move(P1) and not node.board.winning_move(P2):
            node = node.expand()
        res = node.rollout()
        node.backprop(res)
    move, _ = max(root.children.items(), key=lambda it: it[1].visits)
    return move

# ————————————————————————
# AI move wrapper with timing + metrics
# ————————————————————————
def get_ai_move(board, piece, mode, diff, param):
    if mode == 'minimax':
        global minimax_counter
        minimax_counter = 0
        start = time.time()
        col, _ = minimax(board, param, True)
        nodes = minimax_counter
    elif mode == 'alphabeta':
        global alphabeta_counter
        alphabeta_counter = 0
        start = time.time()
        col, _ = alphabeta(board, param, -math.inf, math.inf, True)
        nodes = alphabeta_counter
    elif mode == 'mcts':
        global mcts_counter
        start = time.time()
        col = mcts(board, param)
        nodes = mcts_counter
    else:
        start = time.time()
        col = random.choice(board.valid_moves())
        nodes = 0

    elapsed = time.time() - start

    # compute heuristic & delta for this same 'piece'
    newb = board.copy()
    newb.drop_piece(col, piece)
    h = newb.evaluate(piece)
    best = -math.inf
    for c in board.valid_moves():
        b2 = board.copy()
        b2.drop_piece(c, piece)
        sc = b2.evaluate(piece)
        if sc > best:
            best = sc
    d = best - h

    metrics['nodes'][piece].append(nodes)
    metrics['times'][piece].append(elapsed)
    metrics['heuristic'][piece].append(h)
    metrics['delta'][piece].append(d)

    print(f"[AI][{mode}] P{piece} nodes={nodes} time={elapsed:.3f}s h={h:.1f} Δ={d:.1f} → {col}")
    return col

# ————————————————————————
# UI Helpers (unchanged)
# ————————————————————————
def draw_button(surface, rect, text, font, base_color, hover_color):
    mx, my = pygame.mouse.get_pos()
    hovered = rect.collidepoint(mx, my)
    color   = hover_color if hovered else base_color
    scale   = 1.05 if hovered else 1.0
    w, h    = rect.size
    surf    = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0,0,w,h), border_radius=8)
    surf = pygame.transform.smoothscale(surf, (int(w*scale), int(h*scale)))
    dx, dy = (surf.get_width()-w)//2, (surf.get_height()-h)//2
    surface.blit(surf, (rect.x-dx, rect.y-dy))
    txt = font.render(text.upper(), True, BLACK)
    surface.blit(txt, txt.get_rect(center=(rect.centerx, rect.centery)))
    return hovered

def fade(screen, w, h, fade_in=True, speed=5):
    fade_surf = pygame.Surface((w,h))
    fade_surf.fill(BLACK)
    clock = pygame.time.Clock()
    if fade_in:
        for alpha in range(255, -1, -speed):
            fade_surf.set_alpha(alpha)
            screen.blit(fade_surf, (0,0))
            pygame.display.update()
            clock.tick(FPS)
    else:
        for alpha in range(0, 256, speed):
            fade_surf.set_alpha(alpha)
            screen.blit(fade_surf, (0,0))
            pygame.display.update()
            clock.tick(FPS)

def animate_drop(screen, before_board, after_board, col, piece):
    for r in range(ROWS):
        if after_board.board[r][col] == piece:
            target = r
            break
    x = col*SQUARESIZE + SQUARESIZE//2
    y = SQUARESIZE//2
    clock = pygame.time.Clock()
    while True:
        draw_board(screen, before_board.board)
        pygame.draw.circle(
            screen,
            RED if piece==P1 else YELLOW,
            (x, y+SQUARESIZE),
            RADIUS
        )
        pygame.display.update()
        y += 20
        if (y+SQUARESIZE//2) >= ((target+1)*SQUARESIZE + SQUARESIZE//2):
            break
        clock.tick(FPS)
    draw_board(screen, after_board.board)

def draw_board(screen, board):
    for c in range(COLUMNS):
        for r in range(ROWS):
            pygame.draw.rect(
                screen, BLUE,
                (c*SQUARESIZE, (r+1)*SQUARESIZE,
                 SQUARESIZE, SQUARESIZE)
            )
            color = BLACK
            if board[r][c] == P1:   color = RED
            elif board[r][c] == P2: color = YELLOW
            pygame.draw.circle(
                screen, color,
                (c*SQUARESIZE+SQUARESIZE//2,
                 (r+1)*SQUARESIZE+SQUARESIZE//2),
                RADIUS
            )
    pygame.display.update()

def choose_players_ui(screen):
    font    = pygame.font.SysFont("Arial", 36)
    options = ['human','minimax','alphabeta','mcts']
    diffs   = ['easy','hard']
    params_minimax = ['1','3','5','7']
    params_mcts    = ['100','1000','5000']

    sel = {1:{}, 2:{}}
    stage, player = 1, 1
    clock = pygame.time.Clock()

    while player <= 2:
        screen.fill(WHITE)
        if stage == 1:
            title, choices = f"Joueur {player} : MODE", options
        elif stage == 2:
            title, choices = f"Joueur {player} : DIFFICULTÉ", diffs
        else:
            title = f"Joueur {player} : PARAM"
            choices = params_minimax if sel[player]['mode'] in ('minimax','alphabeta') else params_mcts

        title_s = font.render(title, True, BLACK)
        screen.blit(title_s, title_s.get_rect(center=(screen.get_width()//2, 60)))

        total_w = len(choices)*160 + (len(choices)-1)*20
        x0 = (screen.get_width()-total_w)//2
        y0 = screen.get_height()//2 - 25
        buttons = []
        for i,opt in enumerate(choices):
            rect = pygame.Rect(x0 + i*180, y0, 160, 50)
            hov = draw_button(screen, rect, opt, font, GRAY, HOVER)
            buttons.append((rect,opt,hov))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for rect,opt,hov in buttons:
                    if hov:
                        if stage==1:
                            sel[player]['mode']=opt
                            if opt=='human':
                                sel[player]['diff']=sel[player]['param']=None
                                player+=1; stage=1
                            else:
                                stage=2
                        elif stage==2:
                            sel[player]['diff']=opt; stage=3
                        else:
                            sel[player]['param']=int(opt.split()[0])
                            player+=1; stage=1
                        pygame.time.wait(150)
        clock.tick(FPS)

    return (
        sel[1]['mode'], sel[1]['diff'], sel[1]['param'],
        sel[2]['mode'], sel[2]['diff'], sel[2]['param']
    )

# ————————————————————————
# MAIN
# ————————————————————————
if __name__ == '__main__':
    pygame.init()
    clock = pygame.time.Clock()
    w, h = COLUMNS*SQUARESIZE, (ROWS+1)*SQUARESIZE
    screen = pygame.display.set_mode((w,h))
    pygame.display.set_caption("Sélection des joueurs")
    fade(screen, w, h, fade_in=False)
    p1m,p1d,p1p, p2m,p2d,p2p = choose_players_ui(screen)
    fade(screen, w, h, fade_in=True)
    pygame.display.set_caption("Puissance 4")
    try:
        hwnd = pygame.display.get_wm_info()['window']
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0002|0x0001)
        ctypes.windll.user32.ShowWindow(hwnd, 9)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except:
        pass

    font = pygame.font.SysFont("Arial",75)
    game = ConnectFour()
    turn = random.choice([1,2])
    draw_board(screen, game.board)
    game_over = False

    # clear metrics
    for key in metrics:
        metrics[key][P1].clear(); metrics[key][P2].clear()

    while not game_over:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type==pygame.MOUSEMOTION and (
               (turn==1 and p1m=='human') or (turn==2 and p2m=='human')
            ):
                pygame.draw.rect(screen, BLACK, (0,0,w,SQUARESIZE))
                x = ev.pos[0]
                pygame.draw.circle(screen,
                    RED if turn==1 else YELLOW, (x,SQUARESIZE//2), RADIUS)
                pygame.display.update()

            # human move
            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and (
               (turn==1 and p1m=='human') or (turn==2 and p2m=='human')
            ):
                col = ev.pos[0]//SQUARESIZE
                if col in game.valid_moves():
                    piece = P1 if turn==1 else P2
                    # record human metrics
                    newb = game.copy()
                    newb.drop_piece(col, piece)
                    h = newb.evaluate(piece)
                    best = -math.inf
                    for c in game.valid_moves():
                        b2 = game.copy()
                        b2.drop_piece(c, piece)
                        sc = b2.evaluate(piece)
                        if sc > best: best = sc
                    d = best - h
                    metrics['nodes'][piece].append(0)
                    metrics['times'][piece].append(0.0)
                    metrics['heuristic'][piece].append(h)
                    metrics['delta'][piece].append(d)
                    print(f"[HUMAN] P{piece} move={col} h={h:.1f} Δ={d:.1f}")

                    prev = game.copy()
                    game.drop_piece(col, piece)
                    animate_drop(screen, prev, game, col, piece)
                    if game.winning_move(piece):
                        print(f"[RESULT] Player {turn} wins")
                        game_over = True
                    turn = 2 if turn==1 else 1

        # AI move
        if not game_over and (
           (turn==1 and p1m!='human') or (turn==2 and p2m!='human')
        ):
            piece = P1 if turn==1 else P2
            mode, diff, param = (p1m,p1d,p1p) if turn==1 else (p2m,p2d,p2p)
            prev = game.copy()
            col = get_ai_move(game, piece, mode, diff, param)
            game.drop_piece(col, piece)
            animate_drop(screen, prev, game, col, piece)
            if game.winning_move(piece):
                print(f"[RESULT] P{piece} ({mode}) wins")
                game_over = True
            turn = 2 if turn==1 else 1

        draw_board(screen, game.board)

        if game.is_full() and not game_over:
            print("[RESULT] Draw")
            game_over = True

        clock.tick(FPS)

    pygame.quit()

    # ————————————————————————
    # Display metrics graphs
    # ————————————————————————
    # 2D line plots for nodes & times
    fig1, ax1 = plt.subplots()
    ax1.plot(metrics['nodes'][P1],  color='red',    marker='o', label='Joueur 1')
    ax1.plot(metrics['nodes'][P2],  color='yellow', marker='o', label='Joueur 2')
    ax1.set_title("Nœuds explorés par coup")
    ax1.set_xlabel("Index du coup")
    ax1.set_ylabel("Nœuds explorés")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(metrics['times'][P1],  color='red',    marker='x', label='Joueur 1')
    ax2.plot(metrics['times'][P2],  color='yellow', marker='x', label='Joueur 2')
    ax2.set_title("Temps de réponse par coup (s)")
    ax2.set_xlabel("Index du coup")
    ax2.set_ylabel("Temps (s)")
    ax2.legend()

    # 3D bar for heuristic, only if data exists
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    dx = dy = 0.4
    if metrics['heuristic'][P1]:
        xs1 = list(range(len(metrics['heuristic'][P1])))
        ys1 = [1]*len(xs1)
        zs1 = metrics['heuristic'][P1]
        ax3.bar3d(xs1, ys1, [0]*len(xs1), dx, dy, zs1,
                  color='red', alpha=0.7, label='Joueur 1')
    if metrics['heuristic'][P2]:
        xs2 = list(range(len(metrics['heuristic'][P2])))
        ys2 = [2]*len(xs2)
        zs2 = metrics['heuristic'][P2]
        ax3.bar3d(xs2, ys2, [0]*len(xs2), dx, dy, zs2,
                  color='yellow', alpha=0.7, label='Joueur 2')
    ax3.set_title("Score heuristique par coup (3D)")
    ax3.set_xlabel("Index du coup")
    ax3.set_ylabel("Joueur")
    ax3.set_zlabel("Score heuristique")
    ax3.set_yticks([1,2])
    ax3.set_yticklabels(['J1','J2'])
    ax3.legend()

    # Scatter + line for delta
    fig4, ax4 = plt.subplots()
    ax4.scatter(range(len(metrics['delta'][P1])), metrics['delta'][P1],
                c='red',    marker='s', label='Joueur 1')
    ax4.scatter(range(len(metrics['delta'][P2])), metrics['delta'][P2],
                c='yellow', marker='s', label='Joueur 2')
    ax4.plot(range(len(metrics['delta'][P1])), metrics['delta'][P1],
             color='red', linestyle='--')
    ax4.plot(range(len(metrics['delta'][P2])), metrics['delta'][P2],
             color='yellow', linestyle='--')
    ax4.set_title("Δ heuristique par coup")
    ax4.set_xlabel("Index du coup")
    ax4.set_ylabel("Δ heuristique")
    ax4.legend()

    plt.show()
