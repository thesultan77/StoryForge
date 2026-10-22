# StoryForge

**Collaborative On-Chain Storytelling with an AI Editor, Rolling-Summary Memory and Direct Tipping on GenLayer**

A shared story that anyone can add to, one line at a time. An on-chain LLM
editor accepts only the lines that fit the story and rejects the ones that
break it, then rewrites a short rolling summary so the story can keep growing
without the editor's prompt growing with it. Readers can tip any line, and the
tip goes straight to that line's author.

---

## What it does

1. Anyone can **start a story** with a title, a genre, an opening premise and
   a maximum number of lines (3 to 30). The premise is embedded and compared
   with every existing premise, so the same story idea cannot be started
   twice.
2. Anyone can **add a line** (20 to 400 characters). Each wallet may have up to
   **3 accepted lines per story**, so no single author can take over a story.
3. The contract asks an LLM editor whether the line **fits**: it must continue
   the story coherently, stay in genre, not contradict established facts and
   move the story forward. GenLayer's validator consensus
   (`gl.eq_principle.prompt_comparative`) ensures multiple independent LLM
   runs agree on the categorical `FITS`/`BREAKS` verdict before it is accepted
   on-chain.
4. On `FITS`, the editor also returns an **updated summary** of the whole
   story. The validator-agreed summary replaces the stored one, and the line is
   appended with its author recorded. On `BREAKS`, the line is rejected and
   the author's rejected count goes up.
5. The story becomes `COMPLETE` when it reaches its maximum number of lines,
   or when its starter finishes it early.
6. Anyone can **tip a line**. The tip is forwarded in full to that line's
   author in the same transaction, and tracked per line and per author.
7. Anyone can read the full story, inspect any line, check an author's stats,
   or search for stories with a similar premise.

## Why this is different

Most contracts in this series settle money between two parties. StoryForge is
a shared creative artifact that many different wallets build together, with
the AI acting as an editor rather than a judge of a dispute.

Its core idea is **rolling-summary memory**. Instead of feeding the entire
story back to the model on every line, the editor sees only the premise, a
compressed summary and the last three lines. After every accepted line it
rewrites the summary, and validators agree on that new summary through the
same consensus step that agrees on the verdict. The prompt stays a constant
size whether the story has 3 lines or 30.

The money side is deliberately minimal: tips are forwarded immediately to the
author, so the contract never holds a pool and there is no balance that could
be drained or mis-split.

## Architecture

| Concern | Mechanism |
|---|---|
| Story state | Parallel `TreeMap[str, ...]` fields per story ID (starter, title, premise, genre, rolling summary, max lines, line count, status) |
| Line state | `TreeMap`s keyed by `story_id:line_no` for text, author and tips received |
| Author state | `TreeMap`s keyed by lowercased address for accepted, rejected and tips-received counts, plus `story_id:address` for each author's line count in a story |
| AI editing | `gl.nondet.exec_prompt` inside a closure, reconciled across validators via `gl.eq_principle.prompt_comparative`; one call returns the verdict and the updated summary together |
| Output safety | Editor output is validated (JSON parse, structure, field presence, type, domain of `verdict`, and a non-empty `new_summary` of at most 800 characters when the line fits) before any state mutation |
| Prompt-injection hardening | The contributor's line is passed to the editor explicitly labeled as untrusted text to be judged, never instructions |
| Context compression | Rolling summary: the editor sees premise + summary + last 3 lines only, and rewrites the summary after every accepted line |
| Duplicate premises | Each premise is embedded (`all-MiniLM-L6-v2`, 384-dim) and stored in a `VecDB`; a nearest-neighbour distance under `PREMISE_DUP_THRESHOLD = 0.10` blocks a new story |
| Anti-takeover | At most `MAX_LINES_PER_AUTHOR = 3` accepted lines per wallet per story; rejected lines do not count against the cap |
| Tips | `tip_line` forwards `gl.message.value` to the stored line author in the same transaction, so the contract never holds a balance |

Address-derived lookup keys are built with `str(gl.message.sender_address).lower()`
on the write side and `address.lower()` on the read side, because GenLayer's
`Address` string form is checksummed (mixed-case). No `Address`-typed method
parameters are used anywhere; line authors are only ever captured from
`gl.message.sender_address`.

## Contract methods

### Write

- **`start_story(title: str, premise: str, genre: str, max_lines: int) -> str`**
  Opens a story. Fails if the premise is too short or too similar to an
  existing one. Returns the new `story_id` (e.g. `"S0"`).
- **`add_line(story_id: str, text: str) -> str`**
  Runs the AI editor. Returns `"ACCEPTED - line <n> added. ..."` or
  `"REJECTED - ..."`. Reverts if the story is closed, the line length is out
  of range, or the wallet has used its 3 accepted lines.
- **`finish_story(story_id: str)`**
  Starter-only. Marks the story `COMPLETE` early (needs at least one line).
- **`tip_line(story_id: str, line_no: int)`** *(payable)*
  Forwards the sent value to the line's author.

### View

- **`get_story(story_id: str) -> dict`**
  Starter, title, genre, premise, current rolling summary, line count, max
  lines and status.
- **`read_story(story_id: str) -> str`**
  The title, premise and every accepted line, numbered.
- **`get_line(story_id: str, line_no: int) -> dict`**
  A line's text, author and tips received.
- **`get_author_stats(address: str) -> dict`**
  Lines accepted, lines rejected and total tips received for an address.
- **`find_similar_stories(text: str, top_n: int) -> list[dict]`**
  Semantic search over story premises.
- **`get_story_count() -> int`**

## Tested end-to-end in GenLayer Studio

- `start_story` created `"S0"` (premise about a lighthouse keeper, genre
  `mystery`, 4 lines).
- **Fitting line accepted:** a line about wet footprints on the lamp-room
  staircase was returned as `ACCEPTED` (line 1).
- **Off-topic line rejected:** a line about a stock market crash and a pizza
  party was returned as `REJECTED`, and did not use up the author's cap.
- A second fitting line (the fogged lens) was `ACCEPTED` (line 2).
- `read_story` returned the title, premise and both accepted lines, and
  `get_story` showed the rolling summary **rewritten** to include the
  footprints, the bolted door and the fogged lens, with `line_count` rising as
  lines were accepted.
- `tip_line` forwarded 1 GEN to the line's author, and `get_author_stats`
  confirmed `lines_accepted: "2"`, `lines_rejected: "1"` and
  `tips_received: "1000000000000000000"`.
- **Duplicate premise blocked:** starting a story with the identical premise
  correctly failed with `"A very similar story premise already exists (S0)"`.
- **Personal cap enforced:** after a third accepted line, a fourth attempt
  from the same wallet correctly failed with
  `"You have reached your personal line limit for this story"`.
- `finish_story` closed the story, and `get_story` confirmed
  `status: "COMPLETE"`, `line_count: "3"`, `max_lines: "4"`.
- All test transactions reached `FINALIZED`/`SUCCESS` with validator
  agreement.

## Deployment

```
# v0.2.16
# { "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
# ] }
```

Requires the `genlayer-embeddings` dependency for `VecDB` and
`SentenceTransformer`. Constructor takes no arguments. Deploy directly in
GenLayer Studio or via `genlayer-js` against Studionet / Bradbury testnet.

## Notes on GenVM quirks discovered while building this

- Producing the verdict and the next summary in the **same**
  `eq_principle`-reconciled call means validators agree on both together. The
  summary that gets stored is the one the validators already accepted, so every
  node ends up with identical storage.
- Returning a string from a write method (instead of raising) is how a
  non-fatal outcome such as `REJECTED` is reported without reverting, which is
  what lets the rejected counter persist. Raising or asserting reverts the
  whole transaction, which is what keeps storage consistent when the editor
  output is malformed.
- A small `try`/`except` helper is used for reading possibly-missing
  `TreeMap` counters, so an author who has never been counted reads as zero
  without relying on dict-style `.get()`.
- Building a lookup key inside an f-string index expression is avoided in favor
  of computing the key on its own line first, which keeps the code unambiguous
  across Python versions.

## Notes on design decisions

- The editor's output is validated (JSON structure, field presence, type, and
  the domain of the `verdict` field) before any state change, the same safety
  pattern applied across every contract in this series after it was flagged as
  a required fix on an earlier contract (SemanticDisputeRouter).
- Tips are forwarded immediately instead of being pooled. Pooling tips and
  splitting them later would have required Sybil-resistant weighting, whereas
  immediate forwarding keeps the contract free of any held balance.
- The per-author cap counts only accepted lines, so being rejected costs an
  author nothing beyond the transaction itself.
- Duplicate-premise detection is a similarity check, not a guarantee of
  originality: a sufficiently different retelling of the same idea will pass.

## License

MIT
