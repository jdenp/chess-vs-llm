"""chess-vs-llm: play chess against a local LLM over a tiny web UI.

stdlib + python-chess only. See README.md.
"""
import json
import os
import random
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    CFG = json.load(f)

PORT = int(CFG.get("port", 8090))
BASE_URL = CFG.get("base_url", "http://127.0.0.1:8082").rstrip("/")
API_KEY = CFG.get("api_key", "")
MODEL = CFG.get("model", "")  # empty = first model from /v1/models
MAX_CTX = int(CFG.get("max_context", 131072))
BUDGET = int(MAX_CTX * 0.20)  # token cap, estimated as chars/4
RETRIES = 3  # illegal LLM moves get fed back before falling back to random
MAX_TOKENS = 8192  # leaves room for thinking + answer
LLM_TIMEOUT = 300  # seconds, thinking 27B can be slow

lock = threading.Lock()
state = {}
snapshot = {}

SYSTEM = """You are playing chess against a human. You play black, the human plays white.
Board format: 8 rows for ranks 8 down to 1, 8 columns for files a to h.
White pieces are uppercase (P N B R Q K), black pieces are lowercase (p n b r q k), '.' is empty.
Reply with EXACTLY two lines and nothing else:
MOVE: <one move in UCI notation, e.g. e7e5; when promoting write e7e8q>
COMMENT: <one short sentence of game commentary. Never reveal your plan and never hint the human about the game>"""

MOVE_RE = re.compile(r"MOVE:\s*([A-Za-z0-9=]+)", re.I)
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
COMMENT_RE = re.compile(r"COMMENT:\s*(.+)")


def clean_text(text):
    t = text.strip()
    t = re.sub(r"^```[^\n]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return " ".join(t.split())


def board_ascii(b):
    lines = []
    for rank in range(8, 0, -1):
        cells = []
        for file in range(8):
            p = b.piece_at(chess.square(file, rank - 1))
            cells.append(p.symbol() if p else ".")
        lines.append(f"{rank} " + " ".join(cells))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def san_tail(b, n=12):
    b2 = chess.Board()
    sans = []
    for m in b.move_stack:
        sans.append(b2.san(m))
        b2.push(m)
    tail = sans[-n:]
    offset = len(sans) - len(tail)
    out = []
    for i, s in enumerate(tail):
        mv = offset + i + 1
        out.append(f"{(mv + 1) // 2}.{s}" if mv % 2 == 1 else s)
    return " ".join(out)


def turn_prompt(b):
    n = len(b.move_stack) // 2 + 1
    return (f"Black to move (move {n}).\nBoard:\n{board_ascii(b)}\n"
            f"Moves so far: {san_tail(b)}\n"
            f"Reply with exactly the two lines MOVE and COMMENT.")


def est_tokens(msgs):
    return sum(len(m["content"]) for m in msgs) // 4


def trim(msgs):
    # drop oldest history until under budget; system and live prompt stay
    while len(msgs) > 2 and est_tokens(msgs) > BUDGET:
        msgs.pop(1)
    return msgs


def _is_promotion(b, tok):
    frm, to = chess.parse_square(tok[:2]), chess.parse_square(tok[2:])
    p = b.piece_at(frm)
    if p is None or p.piece_type != chess.PAWN:
        return False
    to_rank = (to >> 3) + 1
    return (p.color == chess.WHITE and to_rank == 8) or (p.color == chess.BLACK and to_rank == 1)


def parse_move(text, b):
    m = MOVE_RE.search(text)
    if not m:
        return None
    tok = m.group(1)
    if UCI_RE.match(tok):
        # bare e7e8 defaults to queen, like Board.push
        uci = tok + "q" if len(tok) == 4 and _is_promotion(b, tok) else tok
        try:
            mv = chess.Move.from_uci(uci)
            if mv in b.legal_moves:
                return mv
        except Exception:
            pass
    try:
        mv = b.parse_san(tok)
        if mv in b.legal_moves:
            return mv
    except Exception:
        pass
    return None


def parse_comment(text):
    m = COMMENT_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()[:200]
    lines = [l.strip() for l in clean_text(text).splitlines()
             if l.strip() and not l.strip().upper().startswith("MOVE")]
    return lines[0][:200] if lines else "(no comment)"


_model = None


def model_name():
    global _model
    if _model is None:
        _model = MODEL
        if not MODEL:
            try:
                data = _http("/v1/models")
                items = data.get("data") or data.get("models") or []
                _model = items[0].get("id") if items else ""
            except Exception:
                _model = "(llm unreachable)"
    return _model


def _http(path, body=None):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(BASE_URL + path, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        return json.loads(r.read().decode())


def llm_chat(messages):
    """Returns (content, reasoning)."""
    body = json.dumps({"model": model_name() or "default",
                       "messages": messages,
                       "temperature": 0.8,
                       "max_tokens": MAX_TOKENS,
                       "chat_template_kwargs": {"reasoning_effort": "medium"}}).encode()
    data = _http("/v1/chat/completions", body)
    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return msg.get("content") or "", reasoning


def llm_turn(b):
    """Ask the LLM for a move. Returns (move, comment, reasoning). Never returns an illegal move."""
    last_err = None
    user_msg = turn_prompt(b)
    msgs = trim([{"role": "system", "content": SYSTEM}] + state["hist"]
                + [{"role": "user", "content": user_msg}])
    for _ in range(RETRIES):
        try:
            content, reasoning = llm_chat(msgs)
        except Exception as e:
            last_err = e
            break
        move = parse_move(content, b)
        if move is not None:
            state["hist"].append({"role": "user", "content": user_msg})
            state["hist"].append({"role": "assistant",
                                  "content": clean_text(content)[:500] or "(no comment)"})
            return move, parse_comment(content), reasoning
        token = MOVE_RE.search(content)
        token = token.group(1) if token else "(none found)"
        msgs.append({"role": "assistant", "content": clean_text(content)[:500]})
        msgs.append({"role": "user",
                     "content": f"Your move '{token}' is not legal from this position. "
                                f"Reply again with exactly the two lines MOVE and COMMENT."})
    m = random.choice(list(b.legal_moves))
    note = f" (llm call failed: {last_err})" if last_err else ""
    return m, f"so many illegal moves, going random{note}", ""


def game_result(b):
    if not b.is_game_over():
        return None
    if b.is_checkmate():
        winner, reason = ("b" if b.turn == chess.WHITE else "w"), "checkmate"
    elif b.is_stalemate():
        winner, reason = None, "stalemate"
    elif b.is_threefold_repetition():
        winner, reason = None, "threefold repetition"
    elif b.is_insufficient_material():
        winner, reason = None, "insufficient material"
    else:
        winner, reason = None, "fifty-move rule"
    return {"winner": winner, "reason": reason}


def closing_comment(b, res):
    if res["winner"] == "b":
        outcome = "The LLM (black) wins."
    elif res["winner"] == "w":
        outcome = "The human (white) wins."
    else:
        outcome = "It is a draw."
    prompt = (f"Game over: {res['reason']}. {outcome} "
              f"Write one short closing comment about the game. No MOVE line.")
    try:
        text, reasoning = llm_chat([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}])
        return clean_text(text)[:300], reasoning
    except Exception as e:
        return f"(llm unreachable: {e})", ""


def pub_state():
    b = state["board"]
    return {
        "fen": b.fen(),
        "turn": "w" if b.turn == chess.WHITE else "b",
        "game_over": b.is_game_over(),
        "result": state["result"],
        "in_check": b.is_check(),
        "legal": [m.uci() for m in b.legal_moves]
                 if not b.is_game_over() and b.turn == chess.WHITE else [],
        "last_move": state["last_move"],
        "thinking": state["thinking"],
        "chat": state["chat"],
        "model": model_name(),
    }


def new_game():
    global state, snapshot
    state = {"board": chess.Board(),
             "chat": [{"who": "sys", "text": "You play white, the LLM plays black. Good luck."}],
             "hist": [], "result": None, "last_move": None, "thinking": False}
    snapshot = pub_state()


def handle_move(data):
    global snapshot
    with lock:
        b = state["board"]
        if b.is_game_over():
            return 409, {"error": "game over, start a new game"}
        if b.turn != chess.WHITE:
            return 409, {"error": "not your turn"}
        frm, to = data.get("from", ""), data.get("to", "")
        promo = (data.get("promotion") or "").lower()
        if promo and promo not in "qrbn":
            return 400, {"error": "bad promotion"}
        try:
            m = chess.Move.from_uci(frm + to)
        except TypeError:
            return 400, {"error": "bad move format"}
        if promo:
            m.promotion = promo
        try:
            b.push(m)
        except (chess.IllegalMoveError, AssertionError):
            # IllegalMoveError = move into check, AssertionError = not even pseudo-legal
            return 400, {"error": f"illegal move: {frm}{to}"}
        state["last_move"] = {"from": frm, "to": to}
        state["thinking"] = True
        snapshot = pub_state()

        res = game_result(b)
        if res is None:
            m2, comment, reasoning = llm_turn(b)
            b.push(m2)
            state["last_move"] = {"from": chess.square_name(m2.from_square),
                                  "to": chess.square_name(m2.to_square)}
            state["chat"].append({"who": "llm", "text": comment,
                                  "reasoning": reasoning or None})
            res = game_result(b)
        if res is not None:
            state["result"] = res
            headline = {"w": "You win!", "b": "LLM wins!"}.get(res["winner"], "Draw.")
            state["chat"].append({"who": "sys",
                                  "text": f"{headline} ({res['reason']})"})
            closing, closing_reason = closing_comment(b, res)
            if closing:
                state["chat"].append({"who": "llm", "text": closing,
                                      "reasoning": closing_reason or None})
        state["thinking"] = False
        snapshot = pub_state()
        return 200, snapshot


class Handler(BaseHTTPRequestHandler):
    server_version = "chess-vs-llm/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        with open(os.path.join(HERE, "index.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._html()
        elif self.path == "/api/state":
            self._send(200, snapshot)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        if self.path == "/api/new":
            with lock:
                new_game()
            self._send(200, snapshot)
        elif self.path == "/api/move":
            code, obj = handle_move(data)
            self._send(code, obj)
        else:
            self._send(404, {"error": "not found"})


def main():
    new_game()
    print(f"chess-vs-llm: http://127.0.0.1:{PORT}  llm: {BASE_URL} ({model_name()})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
