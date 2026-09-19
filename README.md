# chess-vs-llm

Play chess against a local LLM. It is not good at chess - that's the point.
You play white, it plays black, talks after every move, and gets fed back
when it cheats (it will try).

## Run

```
C:\Python313\python.exe server.py
```

Needs `python-chess` (`python -m pip install chess`). Open
http://127.0.0.1:8090.

## Config (config.json)

| key           | meaning                                             |
|---------------|-----------------------------------------------------|
| base_url      | OpenAI-compatible API, e.g. a llama-server          |
| api_key       | optional, empty for a local server                  |
| model         | model name, empty = first from /v1/models           |
| max_context   | model context size in tokens                        |
| port          | web port                                            |

The prompt is capped at 20% of `max_context` (chars/4 estimate). When it
grows past the cap the oldest turns are dropped; the board position is
rebuilt every turn, so dropped history never loses game state.

## How it works

- Your moves are validated by python-chess; illegal clicks just don't happen.
- The LLM must answer with `MOVE: <uci>` and `COMMENT: <line>`. Invalid moves
  get fed back for up to 3 tries, then a random legal move is played.
- Checkmate, stalemate and the draw rules are all enforced server-side.
- The result is fed back to the LLM for a closing comment.
