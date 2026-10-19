#!/usr/bin/env python3
"""Interactive terminal explorer for word2vec binary vector files.

Usage: explore.py [vectors.bin]   (defaults to data/text8-vector.bin)

Query syntax:
  france                  nearest neighbours of a word
  paris france            nearest neighbours of the sum of several words
  king - man + woman      vector arithmetic
  man : king :: woman     analogy (man is to king as woman is to ?)
  cat ~ dog               cosine similarity between two words (or word groups)
  ! breakfast lunch car   odd one out
"""
import curses
import os
import re
import sys
import time

import numpy as np

DEFAULT_VECTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "text8-vector.bin")
BLOCKS = " ▏▎▍▌▋▊▉█"
HELP = "king - man + woman · man : king :: woman · cat ~ dog · ! a b c d · Tab complete · ↑↓ history · Esc quit"
EXAMPLES = [
    ("france", "nearest neighbours"),
    ("king - man + woman", "vector arithmetic"),
    ("paris : france :: tokyo", "analogy"),
    ("coffee ~ tea", "similarity"),
    ("! breakfast lunch dinner car", "odd one out"),
]


class QueryError(Exception):
    pass


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        nl = data.index(b"\n")
        n, dim = map(int, data[:nl].split())
        self.dim = dim
        self.words = []
        self.vecs = np.empty((n, dim), dtype=np.float32)
        pos = nl + 1
        for i in range(n):
            sp = data.index(b" ", pos)
            # Each record is "word " + dim float32s + "\n"; strip the previous record's newline.
            self.words.append(data[pos:sp].decode("utf-8", "replace").strip())
            self.vecs[i] = np.frombuffer(data, dtype=np.float32, count=dim, offset=sp + 1)
            pos = sp + 1 + dim * 4
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vecs /= norms
        self.index = {w: i for i, w in enumerate(self.words)}

    def lookup(self, word):
        for w in (word, word.lower()):
            if w in self.index:
                return self.index[w]
        raise QueryError(f"not in vocabulary: {word}" + self.suggest(word))

    def suggest(self, word):
        word = word.lower()
        for n in range(len(word), 1, -1):
            matches = self.complete(word[:n], 5)
            if matches:
                return "  (try: " + ", ".join(matches) + ")"
        return ""

    def complete(self, prefix, limit):
        # Vocabulary is sorted by frequency, so the first matches are the most common words.
        out = []
        for w in self.words:
            if w.startswith(prefix):
                out.append(w)
                if len(out) == limit:
                    break
        return out

    def nearest(self, v, exclude, k):
        v = v / (np.linalg.norm(v) or 1)
        sims = self.vecs @ v
        sims[list(exclude)] = -np.inf
        k = min(k, len(sims) - len(exclude))
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.words[i], float(sims[i])) for i in top]

    def combine(self, words):
        ids = [self.lookup(w) for w in words]
        return self.vecs[ids].sum(axis=0), set(ids)


def run_query(model, text, k):
    """Return (title, rows) where rows is a list of (word, score)."""
    if ":" in text:
        left, _, right = text.partition("::")
        if left.count(":") != 1 or right.count(":"):
            raise QueryError("analogy syntax: a : b :: c")
        a, b = (s.strip() for s in left.split(":"))
        c = right.strip()
        if not (a and b and c) or " " in a + b + c:
            raise QueryError("analogy syntax: a : b :: c  (one word each)")
        ia, ib, ic = model.lookup(a), model.lookup(b), model.lookup(c)
        v = model.vecs[ib] - model.vecs[ia] + model.vecs[ic]
        return f"{a} is to {b} as {c} is to …", model.nearest(v, {ia, ib, ic}, k)

    if "~" in text:
        parts = [p.split() for p in text.split("~")]
        if len(parts) != 2 or not all(parts):
            raise QueryError("similarity syntax: a ~ b")
        va, _ = model.combine(parts[0])
        vb, _ = model.combine(parts[1])
        sim = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
        return "cosine similarity", [(" ".join(parts[0]) + " ~ " + " ".join(parts[1]), sim)]

    if text.startswith("!"):
        words = text[1:].split()
        if len(words) < 3:
            raise QueryError("odd one out needs at least 3 words")
        ids = [model.lookup(w) for w in words]
        mean = model.vecs[ids].mean(axis=0)
        mean /= np.linalg.norm(mean) or 1
        sims = model.vecs[ids] @ mean
        order = np.argsort(sims)
        return f"odd one out: {words[order[0]]}", [(words[i], float(sims[i])) for i in order]

    v = np.zeros(model.dim, dtype=np.float32)
    exclude = set()
    sign = 1
    terms = 0
    for tok in re.findall(r"[^\s+-]+|[+-]", text):
        if tok in "+-":
            sign = 1 if tok == "+" else -1
            continue
        i = model.lookup(tok)
        v += sign * model.vecs[i]
        exclude.add(i)
        sign = 1
        terms += 1
    if not terms:
        raise QueryError("enter a word")
    return "nearest to " + text, model.nearest(v, exclude, k)


def bar(score, width):
    filled = max(0.0, min(1.0, score)) * width
    full = int(filled)
    s = "█" * full
    if full < width:
        s += BLOCKS[int((filled - full) * 8)]
    return s.ljust(width)


class App:
    def __init__(self, scr, model, name):
        self.scr = scr
        self.model = model
        self.name = name
        self.text = ""
        self.history = []
        self.hist_pos = 0
        self.status = ""
        self.status_err = False
        self.title = "try one of these:"
        self.rows = None
        self.completion = None

        curses.curs_set(1)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)

    def put(self, y, x, s, attr=0):
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                self.scr.addstr(y, x, s[: w - x - (1 if y == h - 1 else 0)], attr)
            except curses.error:
                pass

    def result_capacity(self):
        h, _ = self.scr.getmaxyx()
        return max(1, h - 8)

    def draw(self):
        scr = self.scr
        scr.erase()
        h, w = scr.getmaxyx()
        header = f" word2vec explorer · {self.name} · {len(self.model.words):,} words × {self.model.dim}d "
        self.put(0, 0, header.ljust(w), curses.A_REVERSE | curses.A_BOLD)

        attr = curses.color_pair(3) if self.status_err else curses.color_pair(4)
        self.put(4, 2, self.status, attr)
        self.put(5, 2, self.title, curses.A_BOLD)

        if self.rows is None:
            for n, (q, desc) in enumerate(EXAMPLES):
                self.put(7 + n, 4, q.ljust(32), curses.color_pair(1))
                self.put(7 + n, 36, desc, curses.A_DIM)
        else:
            wordw = min(28, max([len(r[0]) for r in self.rows] + [4]))
            barw = max(5, min(40, w - wordw - 22))
            for n, (word, score) in enumerate(self.rows[: self.result_capacity()]):
                y = 7 + n
                self.put(y, 2, f"{n + 1:>3}", curses.A_DIM)
                self.put(y, 7, word[:wordw].ljust(wordw), curses.color_pair(1))
                self.put(y, 9 + wordw, bar(score, barw), curses.color_pair(2))
                self.put(y, 11 + wordw + barw, f"{score:+.3f}")

        self.put(h - 1, 0, HELP, curses.A_DIM)
        prompt = "› "
        visible = self.text[-(w - 5):]
        self.put(2, 2, prompt, curses.A_BOLD | curses.color_pair(1))
        self.put(2, 2 + len(prompt), visible)
        scr.move(2, min(w - 1, 2 + len(prompt) + len(visible)))
        scr.refresh()

    def submit(self):
        query = self.text.strip()
        if not query:
            return
        if not self.history or self.history[-1] != query:
            self.history.append(query)
        self.hist_pos = len(self.history)
        try:
            start = time.perf_counter()
            self.title, self.rows = run_query(self.model, query, self.result_capacity())
            self.status = f"{(time.perf_counter() - start) * 1000:.0f} ms"
            self.status_err = False
            self.text = ""
        except QueryError as e:
            self.status, self.status_err = str(e), True

    def tab(self):
        c = self.completion
        if c and self.text == c["base"] + c["cands"][c["idx"]]:
            c["idx"] = (c["idx"] + 1) % len(c["cands"])
        else:
            m = re.search(r"[^\s+:~!-]+$", self.text)
            if not m:
                return
            cands = self.model.complete(m.group().lower(), 8)
            if not cands:
                self.status, self.status_err = "no completions", True
                return
            c = self.completion = {"base": self.text[: m.start()], "cands": cands, "idx": 0}
        self.text = c["base"] + c["cands"][c["idx"]]
        self.status = "  ".join(("[" + w + "]") if i == c["idx"] else w for i, w in enumerate(c["cands"]))
        self.status_err = False

    def history_move(self, delta):
        if not self.history:
            return
        self.hist_pos = max(0, min(len(self.history), self.hist_pos + delta))
        self.text = self.history[self.hist_pos] if self.hist_pos < len(self.history) else ""

    def run(self):
        while True:
            self.draw()
            try:
                key = self.scr.get_wch()
            except curses.error:
                continue
            if key != "\t":
                self.completion = None
            if key in ("\n", "\r", curses.KEY_ENTER):
                self.submit()
            elif key == "\t":
                self.tab()
            elif key in ("\x7f", "\b", curses.KEY_BACKSPACE):
                self.text = self.text[:-1]
            elif key == "\x15":  # Ctrl-U
                self.text = ""
            elif key == "\x17":  # Ctrl-W
                self.text = re.sub(r"\S*\s*$", "", self.text)
            elif key == "\x1b":
                if not self.text:
                    return
                self.text = ""
            elif key == "\x04":  # Ctrl-D
                return
            elif key == curses.KEY_UP:
                self.history_move(-1)
            elif key == curses.KEY_DOWN:
                self.history_move(1)
            elif isinstance(key, str) and key.isprintable():
                self.text += key


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VECTORS
    if not os.path.exists(path):
        sys.exit(f"vector file not found: {path}\ntrain one first, e.g. scripts/create-text8-vector-data.sh")
    print(f"loading {path} …", end="", flush=True)
    start = time.perf_counter()
    model = Model(path)
    print(f" {len(model.words):,} words in {time.perf_counter() - start:.1f}s")
    os.environ.setdefault("ESCDELAY", "25")
    try:
        curses.wrapper(lambda scr: App(scr, model, os.path.basename(path)).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
