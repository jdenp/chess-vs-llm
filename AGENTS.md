# chess-vs-llm - notes for agents

- One server (`server.py`, stdlib + python-chess), one page (`index.html`),
  one `config.json`. No build step.
- LLM endpoint is OpenAI-compatible chat completions (llama-server on :8082
  by default).
- Context budget is 20% of config `max_context`, estimated chars/4. Trim
  drops oldest history only, never the system prompt or the live board.
- Never trust the model's move: every LLM move is legality-checked before
  `Board.push`. On failure send the feedback message and retry (max 3),
  then play a random legal move.
- The UI polls `/api/state` every 2s; state changes happen under the
  server lock, snapshots are read lock-free.
- Run: `C:\Python313\python.exe server.py` (port in config.json).
