# hailo10h_chat: an LLM chat in the terminal on a Raspberry Pi 5 with a Hailo-10H

A single Python file that downloads a language model, loads it onto the
Hailo-10H and lets you chat with it in the terminal. The answers stream in
as they are generated. It uses only Python's standard library and HailoRT,
so there is nothing to install with pip.

## What it needs

- A Raspberry Pi 5 with a Hailo-10H.
- HailoRT 5.1.1 and its Python package. On Debian these are the packages
  `h10-hailort` and `python3-h10-hailort`. Other HailoRT versions won't work:
  the models below are compiled for 5.1.1.
- Python 3 from the system (Debian 13 has 3.13), not a virtual environment
  that leaves out the system's packages.
- Internet access the first time a model is used, and room in `~/models`
  (1.7 to 3.4 GB per model).

## Running it

    curl -O https://raw.githubusercontent.com/Arardeth/hailo10h-chat/main/hailo10h_chat.py
    python3 hailo10h_chat.py

The first run downloads the default model (2.4 GB) into `~/models` with a
progress bar, then checks its size and SHA-256. If the download stops, run
the same command again and it continues where it stopped. Loading the model
onto the chip takes a few seconds on every start.

The screen has three parts:

- the conversation, which you can scroll with PgUp and PgDn;
- a status line with the model, the speed of the last answer in tokens per
  second, how much of the model's context is used, and the chip's
  temperature (`-` when it can't be read; above about 90 °C the chat
  warns you to give the chip a break);
- the line where you type.

It works over SSH, including from Windows Terminal or PowerShell, in a
terminal as small as 80x24, and keeps working when you resize the window.

### Keys and commands

| | |
|---|---|
| Enter | send the message |
| Ctrl+C while it answers | stop the answer |
| Ctrl+C at the prompt, Ctrl+D, `/quit` | leave |
| PgUp, PgDn | scroll the conversation |
| Up, Down | recall earlier messages |
| `/clear` | start a new conversation (clears the model's context) |
| `/system TEXT` | set a new system prompt (also clears the context) |
| `/system` | show the system prompt |
| `/help` | list the commands |

The model remembers the conversation until you `/clear` it or its context
is full: the status line shows how full it is, and the chat tells you when
it is nearly full.

### Options

| Option | |
|---|---|
| `--model NAME` | the model to use (see below); default `Qwen2.5-1.5B-Instruct` |
| `--hef PATH` | use a `.hef` file you already have instead (it isn't checked) |
| `--temperature T` | sampling temperature from 0 to 2; 0 always takes the most likely token. Default: the model's own |
| `--max-tokens N` | the most tokens in one answer; default 512 |
| `--system TEXT` | a system prompt for the conversation |
| `--plain` | a plain line-by-line chat instead of the full-screen one, with the same commands. Also used by itself when the input or output isn't a terminal |

For example:

    python3 hailo10h_chat.py --model Qwen2.5-Coder-1.5B-Instruct --system "Answer with code only."
    python3 hailo10h_chat.py --plain --temperature 0

## Models

These are Hailo's builds for HailoRT 5.1.1. They are downloaded from Hailo's
servers (`https://dev-public.hailo.ai/v5.1.1/blob/NAME.hef`) into
`~/models/NAME.hef`. Builds from newer Hailo releases don't load in HailoRT
5.1.1.

| `--model` | Size | Licence of the model |
|---|---|---|
| `Qwen2.5-1.5B-Instruct` (default) | 2.4 GB | Apache-2.0 |
| `Qwen2-1.5B-Instruct` | 1.7 GB | Apache-2.0 |
| `Qwen2.5-Coder-1.5B-Instruct` | 1.8 GB | Apache-2.0 |
| `DeepSeek-R1-Distill-Qwen-1.5B` | 2.4 GB | MIT |
| `Llama-3_2-3B-Instruct` | 3.4 GB | Llama 3.2 Community License (Meta) |

The licences are those of the original models; using a model means
accepting its licence (for Llama 3.2, also Meta's Acceptable Use Policy).
DeepSeek-R1-Distill writes out its reasoning between `<think>` and
`</think>` before the answer, so its answers take longer.

## When something goes wrong

- *Can't import HailoRT's Python package*: install `python3-h10-hailort`
  5.1.1, and run the script with the system's `python3`.
- *Can't open the Hailo device*: only one program at a time can use the
  chip. Stop the other one and try again.
- *Can't load the model*: the `.hef` was compiled for another HailoRT
  version. Use one of the models above.
- HailoRT writes its own log to `hailort.log` in the current directory.
  Messages it prints during the full-screen chat are shown after you leave.

## Licence

The script is under the MIT licence: see [LICENSE](LICENSE). The models
keep their own licences, listed above.
