#!/usr/bin/env python3
"""Chat with a large language model on a Raspberry Pi 5 with a Hailo-10H.

    python3 hailo10h_chat.py [--model NAME | --hef PATH] [--plain]
                             [--temperature T] [--max-tokens N] [--system TEXT]

On first use it downloads the model from Hailo into ~/models and checks its
size and SHA-256, then loads it onto the Hailo-10H and opens a chat in the
terminal. It needs HailoRT 5.1.1 with its Python package
(python3-h10-hailort) and nothing else outside Python's standard library.

In the chat: Enter sends, Ctrl+C stops an answer, Ctrl+C at the prompt,
Ctrl+D or /quit leaves. Commands: /clear, /system TEXT, /help, /quit.
"""

import argparse
import contextlib
import functools
import hashlib
import locale
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

try:
    import curses
except ImportError:  # a Python built without curses: --plain still works
    curses = None

# The models compiled for HailoRT 5.1.1. Those of newer releases of Hailo's
# model zoo don't load in it. Each one's size in bytes and SHA-256.
MODEL_URL = "https://dev-public.hailo.ai/v5.1.1/blob/{name}.hef"
MODELS = {
    "Qwen2.5-1.5B-Instruct": (2359716464, "5310176848638505fbc28add04ba60c97abe345cdb0ec7e3b8ffaa4b0a8c65dd"),
    "Qwen2-1.5B-Instruct": (1678182931, "ab056548c60945cdf4fb30ca43fc7aeed2b9ffc751ad8d4c201dc4c4ab31e86a"),
    "Qwen2.5-Coder-1.5B-Instruct": (1756203793, "88aa7633ebe3385452430ae19f2b459b5a00791cab035576a3262a41ec1350f5"),
    "DeepSeek-R1-Distill-Qwen-1.5B": (2371044685, "9c4506dda44d0a1730d939d4049a3cbf72d5179a88762ca551363db087adb38f"),
    "Llama-3_2-3B-Instruct": (3370416230, "7fc9c7723282482cc3acf08244212668f8b7684deb792b791504ab7f68a83708"),
}
DEFAULT_MODEL = "Qwen2.5-1.5B-Instruct"
MODELS_DIR = Path.home() / "models"

HOT_C = 90  # above about this, the chip needs a break
TEMPERATURE_EVERY_S = 5  # how often the idle chat reads the chip's temperature
CONTEXT_WARN = 0.9  # warn when this share of the context is used


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Chat with an LLM on a Raspberry Pi 5 with a Hailo-10H (HailoRT 5.1.1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models (downloaded from Hailo into ~/models on first use):\n"
        + "".join(f"  {n}{' (default)' if n == DEFAULT_MODEL else ''}\n" for n in MODELS)
        + "\nin the chat: /clear, /system TEXT, /help, /quit; Ctrl+C stops an answer.",
    )
    which = p.add_mutually_exclusive_group()
    which.add_argument("--model", metavar="NAME", choices=list(MODELS), default=DEFAULT_MODEL,
                       help=f"the model to use (default {DEFAULT_MODEL})")
    which.add_argument("--hef", metavar="PATH", help="use this .hef file instead (not checked)")
    p.add_argument("--temperature", type=float, metavar="T",
                   help="sampling temperature, 0 for the most likely token each time (default: the model's own)")
    p.add_argument("--max-tokens", type=int, default=512, metavar="N",
                   help="the most tokens in one answer (default 512)")
    p.add_argument("--system", metavar="TEXT", help="a system prompt")
    p.add_argument("--plain", action="store_true", help="a plain line-by-line chat, without the full-screen UI")
    args = p.parse_args(argv)
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        p.error("--temperature must be between 0 and 2")
    if args.max_tokens < 1:
        p.error("--max-tokens must be at least 1")
    return args


# Downloading a model.

def fetch_model(name, models_dir):
    """The path of model name in models_dir, downloaded and checked if it isn't there yet."""
    size, sha256 = MODELS[name]
    path = models_dir / f"{name}.hef"
    if path.exists():
        have = path.stat().st_size
        if have != size:
            sys.exit(f"{path} has {have} bytes instead of {size}: delete it and run again to download it anew.")
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    free = shutil.disk_usage(models_dir).free
    if free < size - done:
        sys.exit(f"Not enough room in {models_dir}: {name} needs {gb(size - done)} more and {gb(free)} are free.")
    if done:
        print(f"Continuing the download of {name} ({gb(done)} of {gb(size)} already here).")
    else:
        print(f"Downloading {name} ({gb(size)}) from Hailo to {path}")
    try:
        digest = download(MODEL_URL.format(name=name), part, size)
    except KeyboardInterrupt:
        sys.exit("\nDownload stopped. Run the same command again to continue it.")
    except OSError as e:  # urllib's errors are OSErrors too
        sys.exit(f"\nDownload failed: {e}\nRun the same command again to continue it.")
    got = part.stat().st_size
    if got != size or digest != sha256:
        part.unlink()
        sys.exit(f"The download of {name} is not the expected file and was deleted:\n"
                 f"  size    {got} (expected {size})\n  SHA-256 {digest} (expected {sha256})\n"
                 "Run again to download it again.")
    os.replace(part, path)
    print("Size and SHA-256 are right.")
    return path


def download(url, part, size):
    """Downloads url to part, continuing what part already holds; returns the whole file's SHA-256."""
    digest = hashlib.sha256()
    done = part.stat().st_size if part.exists() else 0
    if done == size:
        hash_file(part, digest)
        return digest.hexdigest()
    headers = {"User-Agent": "hailo10h_chat"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
        if done and resp.status == 206:
            hash_file(part, digest)
            mode = "ab"
        else:  # a new download, or the server sent the whole file
            done = 0
            mode = "wb"
        bar = Progress(size, done)
        with open(part, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                bar.show(done)
        bar.show(done, last=True)
    return digest.hexdigest()


def hash_file(path, digest):
    print(f"Checking the {gb(path.stat().st_size)} already downloaded...")
    with open(path, "rb") as f:
        for chunk in iter(functools.partial(f.read, 1 << 20), b""):
            digest.update(chunk)


class Progress:
    """A progress bar on stderr; every 10% on a line of its own when stderr isn't a terminal."""

    def __init__(self, total, start):
        self.total, self.start = total, start
        self.t0 = self.shown = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.tenth = start * 10 // total

    def show(self, done, last=False):
        now = time.monotonic()
        if not last and now - self.shown < 0.25:
            return
        self.shown = now
        rate = (done - self.start) / max(now - self.t0, 1e-3)
        left = f"{clock((self.total - done) / rate)} left" if rate > 0 and done < self.total else ""
        text = f" {done / self.total:4.0%}  {done / 1e9:.2f}/{self.total / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s  {left}"
        if not self.tty:
            tenth = done * 10 // self.total
            if tenth > self.tenth or last:
                self.tenth = tenth
                print(text.strip(), file=sys.stderr, flush=True)
            return
        width = shutil.get_terminal_size((80, 24)).columns - 1
        bar_w = width - len(text) - 2
        if bar_w >= 10:
            filled = bar_w * done // self.total
            text = "[" + "#" * filled + "-" * (bar_w - filled) + "]" + text
        sys.stderr.write("\r" + text[:width] + "\x1b[K" + ("\n" if last else ""))
        sys.stderr.flush()


def gb(n):
    return f"{n / 1e9:.1f} GB"


def clock(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


# The model and the conversation, shared by both user interfaces.

def import_hailo():
    """HailoRT's VDevice and LLM, imported only once the arguments are good."""
    try:
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM
    except Exception as e:  # ImportError, or HailoRT's own version checks
        sys.exit(f"Can't import HailoRT's Python package hailo_platform: {e}\n"
                 "This needs HailoRT 5.1.1 with its Python package (Debian: python3-h10-hailort),\n"
                 "and the system's python3, not a virtual environment without its packages.")
    return VDevice, LLM


class Thermometer:
    """The chip's temperature in °C, or None when it can't be read."""

    def __init__(self, vdevice):
        # Kept in a variable: HailoRT releases a device that isn't, and its
        # control then fails ("the device in use has been released").
        try:
            self.device = vdevice.get_physical_devices()[0]
        except Exception:
            self.device = None
        self.failures = 0

    def read(self):
        if self.device is None or self.failures >= 3:
            return None
        try:
            temp = float(self.device.control.get_chip_temperature().ts0_temperature)
        except Exception:
            self.failures += 1
            return None
        self.failures = 0
        return temp


def quietly(fn):
    try:
        return fn()
    except Exception:
        return None


class Chat:
    def __init__(self, llm, thermometer, name, args):
        self.llm = llm
        self.thermometer = thermometer
        self.name = name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.system = args.system
        self.send_system = bool(self.system)  # with the next message
        self.tokens_per_s = None
        self.context_capacity = quietly(llm.max_context_capacity)
        self.context_used = None
        self.chip_c = None
        self.warned_hot = False
        self.warned_context = False
        self.update_context()
        self.update_temperature()

    def update_context(self):
        self.context_used = quietly(self.llm.get_context_usage_size)

    def update_temperature(self):
        self.chip_c = self.thermometer.read()

    def clear(self):
        self.llm.clear_context()
        self.send_system = bool(self.system)
        self.warned_context = False
        self.update_context()

    def set_system(self, text):
        self.system = text
        self.clear()

    def answer(self, text, on_token, should_stop):
        """Streams the answer to text through on_token(str), asking should_stop() after each token.

        Returns "done", "stopped", or "cut" when the answer reached --max-tokens.
        HailoRT keeps the conversation in the model's context, so only the new
        message is sent (with the system prompt at the start of a conversation).
        """
        messages = []
        if self.send_system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": text})
        params = {"max_generated_tokens": self.max_tokens}
        if self.temperature == 0:
            params["do_sample"] = False
        elif self.temperature is not None:
            params["temperature"] = self.temperature
        result = "done"
        count, first, last = 0, None, None
        # Leaving this block while the model is still answering aborts the
        # answer: HailoRT stops it and closes the turn in the context, so the
        # next message starts clean.
        with self.llm.generate(messages, **params) as gen:
            self.send_system = False
            tokens = iter(gen)
            while True:
                try:
                    token = next(tokens)
                except StopIteration:
                    break
                except UnicodeDecodeError:  # a token that isn't whole UTF-8
                    token = "�"
                last = time.monotonic()
                if first is None:
                    first = last
                count += 1
                on_token(token)
                if should_stop():
                    result = "stopped"
                    break
            if result == "done" and "MAX_TOKENS" in status_name(gen):
                result = "cut"
        if count > 1 and last > first:
            self.tokens_per_s = (count - 1) / (last - first)
        self.update_context()
        return result

    def warnings(self):
        """Warnings to show once: a hot chip, a nearly full context."""
        out = []
        t = self.chip_c
        if t is not None and t >= HOT_C and not self.warned_hot:
            self.warned_hot = True
            out.append(f"The chip is at {t:.0f} °C. Above about {HOT_C} °C it needs a break: "
                       "give it a minute before the next question.")
        elif t is None or t < HOT_C - 5:
            self.warned_hot = False
        used, cap = self.context_used, self.context_capacity
        if used and cap and used >= CONTEXT_WARN * cap and not self.warned_context:
            self.warned_context = True
            out.append(f"The context is nearly full ({used} of {cap} tokens): /clear starts a new conversation.")
        return out

    def status(self):
        tps = f"{self.tokens_per_s:.1f} tok/s" if self.tokens_per_s else "- tok/s"
        used = "-" if self.context_used is None else self.context_used
        cap = "-" if self.context_capacity is None else self.context_capacity
        chip = "-" if self.chip_c is None else f"{self.chip_c:.0f} °C"
        return tps, f"ctx {used}/{cap}", f"chip {chip}"


def status_name(gen):
    try:
        status = gen.generation_status
    except Exception:
        return ""
    return getattr(status, "name", str(status))


def help_text(keys):
    return ("Commands:\n"
            "  /clear        start a new conversation (clears the model's context)\n"
            "  /system TEXT  a new system prompt (also clears the context)\n"
            "  /system       show the system prompt\n"
            "  /help         this help\n"
            "  /quit         leave\n"
            "A message that starts with / is sent with //.\n" + keys)


def run_command(chat, text, keys):
    """Runs the /command text; returns (what, message), what being quit, clear, info or error."""
    parts = text[1:].split(None, 1)
    name = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    if name in ("quit", "exit", "q"):
        return "quit", ""
    if name == "help":
        return "info", help_text(keys)
    if name == "clear":
        chat.clear()
        return "clear", "Context cleared: a new conversation."
    if name == "system":
        if not arg:
            return "info", f"System prompt: {chat.system}" if chat.system else "No system prompt: the model's default."
        chat.set_system(arg)
        return "clear", f"New system prompt, context cleared: {arg}"
    return "error", f"Unknown command /{name}: /help lists them."


def clean(text):
    """text without control characters other than newlines; tabs as spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch) != "Cc")


# The plain chat.

PLAIN_KEYS = "Ctrl+C stops an answer; Ctrl+C or Ctrl+D at the prompt leaves."


def run_plain(chat):
    for stream in (sys.stdin, sys.stdout):
        if (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    print(f"Chatting with {chat.name} on the Hailo-10H. /help lists the commands. {PLAIN_KEYS}")
    if chat.system:
        print(f"System prompt: {chat.system}")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.startswith("/") and not text.startswith("//"):
            try:
                what, message = run_command(chat, text, PLAIN_KEYS)
            except Exception as e:
                what, message = "error", f"Error: {e}"
            if what == "quit":
                return
            print(message)
            continue
        if text.startswith("//"):
            text = text[1:]

        stop = [False]
        started = [False]

        def on_token(token):
            token = clean(token)
            if not started[0]:
                token = token.lstrip()
                started[0] = bool(token)
            sys.stdout.write(token)
            sys.stdout.flush()

        # Ctrl+C asks the answer to stop (after the token being read)
        # instead of raising KeyboardInterrupt somewhere inside HailoRT.
        previous = signal.signal(signal.SIGINT, lambda signum, frame: stop.__setitem__(0, True))
        try:
            result = chat.answer(text, on_token, lambda: stop[0])
        except Exception as e:
            print(f"\nError: {e}\n(/clear may help.)")
            continue
        finally:
            signal.signal(signal.SIGINT, previous)
        print()
        if result == "stopped":
            print("[stopped]")
        elif result == "cut":
            print(f"[cut at --max-tokens {chat.max_tokens}]")
        chat.update_temperature()
        print("[" + " | ".join(chat.status()) + "]")
        for warning in chat.warnings():
            print(warning)


# The full-screen chat.

TUI_KEYS = ("Keys: Enter sends; Ctrl+C stops an answer; Ctrl+C or Ctrl+D at the prompt leaves;\n"
            "PgUp/PgDn scroll; Up/Down recall earlier messages; Ctrl+U clears the line.")

CTRL_C, CTRL_D, ESC = "\x03", "\x04", "\x1b"


@functools.lru_cache(maxsize=4096)
def char_width(ch):
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def text_width(text):
    return sum(char_width(ch) for ch in text)


def wrap(text, width):
    """text as lines of at most width columns, broken at spaces where it can."""
    width = max(width, 2)
    lines = []
    for para in text.split("\n"):
        line, w, space = [], 0, -1
        for ch in para:
            cw = char_width(ch)
            if w + cw > width and line:
                if ch == " ":
                    lines.append("".join(line))
                    line, w, space = [], 0, -1
                    continue
                if space > 0:
                    lines.append("".join(line[:space]))
                    line = line[space + 1:]
                else:
                    lines.append("".join(line))
                    line = []
                w, space = text_width(line), -1
            if ch == " ":
                space = len(line)
            line.append(ch)
            w += cw
        lines.append("".join(line))
    return lines


def clip(text, width):
    out, w = [], 0
    for ch in text:
        w += char_width(ch)
        if w > width:
            break
        out.append(ch)
    return "".join(out)


class Entry:
    """A message in the conversation pane, with its lines for the last width."""

    def __init__(self, kind, text=""):
        self.kind, self.text = kind, text
        self.key, self.cached = None, []

    def lines(self, width):
        key = (width, len(self.text))
        if key != self.key:
            if self.kind == "user":
                lines = wrap(self.text, width - 2)
                self.cached = ["> " + lines[0]] + ["  " + line for line in lines[1:]]
            else:
                self.cached = wrap(self.text, width)
            self.key = key
        return self.cached


class TUI:
    PROMPT = "> "

    def __init__(self, scr, chat):
        self.scr, self.chat = scr, chat
        self.entries = []
        self.buf, self.cur, self.hstart = [], 0, 0  # the input line, its cursor, its first shown character
        self.history, self.hist_i, self.saved = [], None, []
        self.scroll, self.last_total = 0, 0  # lines scrolled up from the bottom
        self.pending = []  # keys read ahead
        self.busy = self.stop = self.quit = False
        self.temperature_at = time.monotonic()
        self.colors()

    def colors(self):
        A = curses
        self.attr = {"user": A.A_BOLD, "bot": A.A_NORMAL, "info": A.A_DIM, "error": A.A_BOLD,
                     "status": A.A_REVERSE, "hot": A.A_REVERSE | A.A_BOLD, "prompt": A.A_BOLD}
        if not A.has_colors():
            return
        try:
            A.start_color()
            try:
                A.use_default_colors()
                bg = -1
            except A.error:
                bg = A.COLOR_BLACK
            A.init_pair(1, A.COLOR_CYAN, bg)
            A.init_pair(2, A.COLOR_YELLOW, bg)
            A.init_pair(3, A.COLOR_RED, bg)
            A.init_pair(4, A.COLOR_WHITE, A.COLOR_RED)
        except A.error:
            return
        self.attr.update(user=A.color_pair(1) | A.A_BOLD, info=A.color_pair(2), error=A.color_pair(3) | A.A_BOLD,
                         hot=A.color_pair(4) | A.A_BOLD, prompt=A.color_pair(1) | A.A_BOLD)

    def add(self, kind, text):
        self.entries.append(Entry(kind, clean(text)))
        self.scroll = 0

    def run(self):
        self.add("info", f"Chatting with {self.chat.name} on the Hailo-10H. Enter sends, Ctrl+C stops an answer, "
                         "/help lists the commands.")
        if self.chat.system:
            self.add("info", f"System prompt: {self.chat.system}")
        for warning in self.chat.warnings():
            self.add("error", warning)
        while not self.quit:
            if time.monotonic() - self.temperature_at >= TEMPERATURE_EVERY_S:
                self.read_temperature()
            self.draw()
            key = self.key(1000)
            if key is not None:
                self.handle(key)

    def read_temperature(self):
        self.chat.update_temperature()
        self.temperature_at = time.monotonic()
        for warning in self.chat.warnings():
            self.add("error", warning)

    # Keys.

    def key(self, timeout_ms):
        """The next key (a str, or an int for special keys), or None after timeout_ms."""
        if self.pending:
            return self.pending.pop(0)
        self.scr.timeout(timeout_ms)
        try:
            return self.scr.get_wch()
        except curses.error:
            return None

    def handle(self, k):
        if k == CTRL_C:
            if self.busy:
                self.stop = True
            else:
                self.quit = True
        elif k == CTRL_D:
            self.stop = self.quit = True
        elif k in ("\r", "\n", curses.KEY_ENTER):
            self.enter(k)
        elif k == curses.KEY_RESIZE:
            curses.update_lines_cols()
        elif k == "\x0c":  # Ctrl+L
            self.scr.clearok(True)
        elif k == ESC:
            self.skip_escape()
        elif k in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            step = max(1, self.scr.getmaxyx()[0] - 3)
            self.scroll = max(0, self.scroll + (step if k == curses.KEY_PPAGE else -step))
        elif k in (curses.KEY_UP, curses.KEY_DOWN):
            self.recall(-1 if k == curses.KEY_UP else 1)
        else:
            self.edit(k)

    def enter(self, k):
        # An Enter with more keys right behind it is part of a paste: keep
        # the newline in the message instead of sending half of it.
        nxt = self.key(30)
        if k == "\r" and nxt == "\n":
            nxt = self.key(30)
        if nxt is not None:
            self.pending.insert(0, nxt)
            self.insert("\n")
        elif not self.busy:  # while answering, the line waits
            self.submit()

    def skip_escape(self):
        # An escape sequence curses didn't know (Alt+key, unusual keys): drop it.
        nxt = self.key(0)
        if nxt in ("[", "O"):
            while True:
                k = self.key(0)
                if k is None or (isinstance(k, str) and "@" <= k <= "~"):
                    return
        elif nxt is not None:
            self.pending.insert(0, nxt)

    def edit(self, k):
        b, c = self.buf, self.cur
        if k in ("\x7f", "\x08", curses.KEY_BACKSPACE):
            if c:
                del b[c - 1]
                self.cur -= 1
        elif k == curses.KEY_DC:
            if c < len(b):
                del b[c]
        elif k == curses.KEY_LEFT:
            self.cur = max(0, c - 1)
        elif k == curses.KEY_RIGHT:
            self.cur = min(len(b), c + 1)
        elif k in (curses.KEY_HOME, "\x01"):
            self.cur = 0
        elif k in (curses.KEY_END, "\x05"):
            self.cur = len(b)
        elif k == "\x15":  # Ctrl+U
            del b[:c]
            self.cur = 0
        elif k == "\x0b":  # Ctrl+K
            del b[c:]
        elif k == "\x17":  # Ctrl+W
            i = c
            while i and b[i - 1] == " ":
                i -= 1
            while i and b[i - 1] != " ":
                i -= 1
            del b[i:c]
            self.cur = i
        elif isinstance(k, str) and unicodedata.category(k) != "Cc":
            self.insert(k)

    def insert(self, text):
        self.buf[self.cur:self.cur] = text
        self.cur += len(text)

    def recall(self, step):
        if not self.history:
            return
        if self.hist_i is None:
            if step > 0:
                return
            self.saved, self.hist_i = self.buf, len(self.history)
        self.hist_i += step
        if self.hist_i >= len(self.history):
            self.buf, self.hist_i = self.saved, None
        else:
            self.hist_i = max(0, self.hist_i)
            self.buf = list(self.history[self.hist_i])
        self.cur = len(self.buf)

    def submit(self):
        text = "".join(self.buf).strip()
        self.buf, self.cur, self.hstart, self.hist_i = [], 0, 0, None
        if not text:
            return
        if not self.history or self.history[-1] != text:
            self.history.append(text)
        if text.startswith("/") and not text.startswith("//"):
            try:
                what, message = run_command(self.chat, text, TUI_KEYS)
            except Exception as e:
                what, message = "error", f"Error: {e}"
            if what == "quit":
                self.quit = True
            elif what == "clear":
                self.entries = []
                self.add("info", message)
            else:
                self.add(what, message)
            return
        if text.startswith("//"):
            text = text[1:]
        self.ask(text)

    # Answers.

    def ask(self, text):
        self.add("user", text)
        bot = Entry("bot")
        self.entries.append(bot)
        self.busy, self.stop = True, False
        self.draw()

        def on_token(token):
            token = clean(token)
            if not bot.text:
                token = token.lstrip()
            bot.text += token
            while not self.stop:  # the keys typed meanwhile
                k = self.key(0)
                if k is None:
                    break
                self.handle(k)
            self.draw()

        try:
            result = self.chat.answer(text, on_token, lambda: self.stop)
        except Exception as e:
            result = None
            self.add("error", f"Error: {e} (/clear may help)")
        finally:
            self.busy = False
        if not bot.text:
            self.entries.remove(bot)
        if result == "stopped":
            self.add("info", "[stopped]")
        elif result == "cut":
            self.add("info", f"[cut at --max-tokens {self.chat.max_tokens}]")
        self.read_temperature()

    # Drawing.

    def put(self, y, x, text, attr=0):
        try:
            self.scr.addstr(y, x, text, attr)
        except curses.error:  # the bottom-right cell, or a window shrinking under us
            pass

    def draw(self):
        h, w = self.scr.getmaxyx()
        self.scr.erase()
        if h < 3 or w < 20:
            self.put(0, 0, "Window too small"[:max(w - 1, 0)])
            self.scr.refresh()
            return
        pane_h = h - 2
        lines = []
        for i, e in enumerate(self.entries):
            if i and e.kind == "user":
                lines.append(("", 0))
            attr = self.attr[e.kind]
            lines.extend((line, attr) for line in e.lines(w - 1))
        if self.scroll:  # keep what's on screen in place while lines arrive
            self.scroll += len(lines) - self.last_total
        self.last_total = len(lines)
        self.scroll = max(0, min(self.scroll, len(lines) - pane_h))
        start = max(0, len(lines) - pane_h - self.scroll)
        for row, (line, attr) in enumerate(lines[start:start + pane_h]):
            self.put(row, 0, line, attr)
        self.draw_status(h - 2, w)
        self.draw_input(h - 1, w)
        self.scr.refresh()

    def draw_status(self, y, w):
        tps, ctx, chip = self.chat.status()
        hot = self.chat.chip_c is not None and self.chat.chip_c >= HOT_C
        hint = "Ctrl+C stops" if self.busy else "PgDn: back down" if self.scroll else "/help"
        rest = f" | {tps} | {ctx} | "
        name = self.chat.name
        room = w - len(rest) - len(chip) - len(hint) - 3
        if room < min(text_width(name), 12):  # a narrow window: the hint goes first
            hint = ""
            room = w - len(rest) - len(chip) - 2
        if text_width(name) > room:
            name = clip(name, max(room - 1, 4)) + "…"
        left = " " + name + rest
        self.put(y, 0, " " * w, self.attr["status"])
        self.put(y, 0, clip(left, w), self.attr["status"])
        x = text_width(left)
        if x < w:
            self.put(y, x, clip(chip, w - x), self.attr["hot" if hot else "status"])
        x += len(chip)
        if hint and x + len(hint) + 2 <= w:
            self.put(y, w - len(hint) - 1, hint, self.attr["status"])

    def draw_input(self, y, w):
        prompt = self.PROMPT
        room = w - len(prompt) - 1
        shown = ["↵" if ch == "\n" else ch for ch in self.buf]
        ends = [0]  # ends[i]: the width of shown[:i]
        for ch in shown:
            ends.append(ends[-1] + char_width(ch))
        self.hstart = min(self.hstart, self.cur)
        while ends[self.cur] - ends[self.hstart] > room - 1:
            self.hstart += 1
        visible = clip("".join(shown[self.hstart:]), room)
        self.put(y, 0, prompt, self.attr["prompt"])
        self.put(y, len(prompt), visible)
        try:
            self.scr.move(y, len(prompt) + ends[self.cur] - ends[self.hstart])
        except curses.error:
            pass


@contextlib.contextmanager
def stderr_saved():
    """Sends what HailoRT writes to stderr to a file while the screen is curses's, and shows it after."""
    sys.stderr.flush()
    saved = os.dup(2)
    log = tempfile.TemporaryFile()
    os.dup2(log.fileno(), 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        log.seek(0)
        data = log.read()
        log.close()
        if data:
            with contextlib.suppress(OSError):
                sys.stderr.write("Messages during the chat:\n" + data[-4000:].decode(errors="replace"))
                sys.stderr.flush()


def run_tui(chat):
    os.environ.setdefault("ESCDELAY", "25")
    with stderr_saved():
        scr = curses.initscr()
        try:
            curses.noecho()
            curses.raw()  # Ctrl+C arrives as a key, not as a signal
            scr.keypad(True)
            with contextlib.suppress(curses.error):
                curses.curs_set(1)
            TUI(scr, chat).run()
        finally:
            with contextlib.suppress(curses.error):  # fails when SSH has dropped
                scr.keypad(False)
                curses.noraw()
                curses.echo()
                curses.endwin()


def why_not_tui():
    """Why the full-screen chat can't run here, or None."""
    if curses is None:
        return "this Python has no curses"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "input or output isn't a terminal"
    if os.environ.get("TERM", "dumb") in ("", "dumb"):
        return "TERM isn't set to a terminal type"
    try:
        curses.setupterm()
    except curses.error as e:
        return f"curses can't use this terminal ({e})"
    return None


# Starting up.

TICKER = r"""
import os, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
label, parent, t0 = sys.argv[1], int(sys.argv[2]), time.monotonic()
while os.getppid() == parent:
    sys.stderr.write("\r%s %d s " % (label, time.monotonic() - t0))
    sys.stderr.flush()
    time.sleep(0.5)
"""


@contextlib.contextmanager
def ticking(label):
    """Shows label with the seconds going by. HailoRT holds Python's lock
    while it loads a model, so the counting runs in a small process of its own."""
    t0 = time.monotonic()
    ticker = None
    if sys.stderr.isatty():
        with contextlib.suppress(OSError):
            ticker = subprocess.Popen([sys.executable, "-I", "-S", "-c", TICKER, label, str(os.getpid())])
    else:
        print(label, file=sys.stderr, flush=True)
    try:
        yield
    finally:
        if ticker:
            ticker.kill()
            ticker.wait()
            sys.stderr.write("\r\x1b[K")
        print(f"{label} {time.monotonic() - t0:.1f} s", file=sys.stderr, flush=True)


def setup_locale():
    """A UTF-8 locale, so that curses reads and writes UTF-8 (Turkish letters, too)."""
    for name in ("", "C.UTF-8"):
        try:
            locale.setlocale(locale.LC_ALL, name)
        except locale.Error:
            continue
        if "utf" in locale.nl_langinfo(locale.CODESET).lower():
            return


def exit_on_signal(signum, frame):
    # SIGTERM, or SIGHUP when SSH drops: leave through the finally blocks,
    # which put the terminal back and release the chip.
    sys.exit(128 + signum)


def main():
    setup_locale()
    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)
    args = parse_args()
    if args.hef:
        hef = Path(args.hef).expanduser()
        if not hef.is_file():
            sys.exit(f"{hef}: no such file")
        name = hef.stem
    else:
        hef = fetch_model(args.model, MODELS_DIR)
        name = args.model
    VDevice, LLM = import_hailo()
    plain = args.plain
    if not plain:
        why = why_not_tui()
        if why:
            print(f"A plain chat: {why}.", file=sys.stderr)
            plain = True

    with contextlib.ExitStack() as stack:
        # Both stay open, in with-blocks, for the whole chat.
        try:
            vdevice = stack.enter_context(VDevice())
        except Exception as e:
            sys.exit(f"Can't open the Hailo device: {e}\n"
                     "Is another program using it? Only one at a time can.")
        try:
            with ticking(f"Loading {name} onto the Hailo-10H..."):
                llm = stack.enter_context(LLM(vdevice, str(hef)))
        except Exception as e:
            sys.exit(f"Can't load {hef}: {e}\n"
                     "This needs a model compiled for HailoRT 5.1.1, such as the ones --help lists.")
        chat = Chat(llm, Thermometer(vdevice), name, args)
        if plain:
            run_plain(chat)
        else:
            run_tui(chat)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
