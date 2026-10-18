# 👽 frank

A local coding agent on the DGX Spark. Two parts: a model server on the Spark
with as much context as the box can hold, and `frank`, a terminal agent with
tool calling that works the way Claude Code does. The target is three things
at once: a context window of 500k tokens or more, deep thinking before the
model acts, and reliable tool calls. Everything stays inside this
directory on both machines: tools, the llama.cpp build, weights, logs, the
agent's venv and sessions.

What it is not: a frontier model. On their vendors' own numbers the best
models that fit this box score around 60 on SWE-bench Pro, and the comparison
table on Poolside's Laguna card puts the hosted model this is modelled on at
80. It is slower and less sharp, and the prompts are written to make it
careful.

## Hardware, and what follows from it

The Spark is sold as 128 GB. The system reports 121.7 GiB of unified memory,
at 273 GB/s. Two things follow from those numbers.

Decode speed is roughly bandwidth divided by the bytes of weights read per
token. A sparse model with a few billion active parameters runs well however
large it is in total, and a dense model reads all of itself for every token.
Measured here: 44 to 49 tok/s for a 125B model with 6B active, 12 tok/s for a
dense 27B.

Context comes out of what the weights leave. vLLM takes a fixed share of the
121.7 GiB (`MEM_UTIL` in a profile), loads the weights and gives the rest to
the KV cache, so a model with 93 GiB of weights has room for about 350k tokens
of cache whatever window it was trained for. Reserving more than about 0.80
of the box does not work: at that setting with a large cache the GPU driver
ran out of memory during warm-up and the server hung.

## What was measured

`scripts/context-test.py` is the acceptance test. It fills the prompt with
numbered records ("Record 9337: the access code for locker 9337 is 836244."),
asks for three of the codes, one near the start, one in the middle and one
near the end, and wants them back through a tool call. A size passes when the
call is made and all three codes are right. Thousands of nearly identical
lines make it a harsh test. It is also the property a long window is for.
Each answer gets 32,768 tokens, thinking included, which is what frank
allows. Until 14 October it was 4,000, which cut DeepSeek off mid-thought
once; the models measured before then are queued again at the new budget.

`scripts/bench.py` wraps it. It measures decode speed on a short prompt,
checks that the model thinks, calls a tool and uses the result, and then runs
the recall test at 100k, 250k, 500k, 750k and 1M, or as far as the window
goes. Each size extends the one before, so the server reads only the new part.
`spark/bake.sh` does that for a list of profiles in turn, downloading and
switching as it goes, and appends a line per model to `run/results.jsonl`.
The control page ranks everything measured so far. `scripts/results.py
--publish` folds the runs into `docs/results.json` and rewrites the tables
here and on the project page.

The ranking puts first the largest prompt answered exactly with every smaller
one exact as well, then the largest with at most one code wrong at each size,
then the share of codes right, then decode speed. Sizes count to the nearest
50k, because the same step comes out a few thousand tokens apart from one
tokenizer to the next. All on this Spark, with thinking on where the model
has it:

<!-- results:start -->
| | model | window served | exact up to | codes right, by prompt size | thinks, tools | decode | read-in |
|---|---|---|---|---|---|---|---|
| 1 | DeepSeek V4 Flash (284B, 13B active, 3-bit GGUF, 104 GB) | 857k, llama.cpp, 16-bit cache, reasoning effort max | 805k | 61k 3/3, 255k 3/3, 501k 3/3, 805k 3/3 | yes | 19 tok/s | 325 tok/s |
| 2 | DeepSeek V4 Flash, 2-bit (284B, 13B active, 2-bit GGUF, 97 GB) | 1.05M, llama.cpp, 16-bit cache, reasoning effort max | 501k | 61k 3/3, 255k 3/3, 501k 3/3, 751k 2/3, 985k 3/3 | yes | 20 tok/s | 320 tok/s |
| 3 | Qwen3.8-27B, FP8 cache (dense 27B, NVFP4, 22 GB) | 1.01M, vLLM, stretched from 256k, FP8 cache | 519k | 93k 3/3, 253k 3/3, 519k 3/3, 986k 2/3 | yes | 12 tok/s | 918 tok/s |
| 4 | DeepSeek V4 Flash, REAP 150B (150B of 284B kept, 13B active, FP4 experts, 85 GB) | 1.05M, llama.cpp, 16-bit cache, reasoning effort max | 501k | 61k 3/3, 255k 3/3, 501k 3/3, 751k 2/3, 985k 2/3 | yes | 16 tok/s | 462 tok/s |
| 5 | Qwen3.8-27B, 16-bit cache (dense 27B, NVFP4, 22 GB) | 950k, stretched from 256k, 16-bit cache | 507k | 97k 3/3, 245k 3/3, 507k 3/3 | yes | 12 tok/s | 920 tok/s |
| 6 | Qwen3.5-122B-A10B, FP8 cache (122B, 10B active, NVFP4, 84 GB) | 1.01M, vLLM, stretched from 256k, FP8 cache | 519k | 93k 3/3, 253k 3/3, 519k 3/3, 760k 1/3, 960k 0/3 | yes | 14 tok/s | 1,030 tok/s |
| 7 | DeepSeek V4 Flash, REAP 200B (200B of 284B kept, 13B active, 3-bit GGUF, 93 GB) | 1.05M, llama.cpp, 16-bit cache, reasoning effort max | 255k | 61k 3/3, 255k 3/3, 501k 2/3, 751k 2/3, 985k 3/3 | yes | 17 tok/s | 360 tok/s |
| 8 | Qwen3.6-35B-A3B (35B, 3B active, NVFP4, 23 GB) | 1.01M, vLLM, stretched from 256k, 16-bit cache | 253k | 93k 3/3, 253k 3/3, 519k 2/3, 760k 2/3, 960k 2/3 | yes | 78 tok/s | 2,897 tok/s |
| 9 | Qwen3.5-122B-A10B (122B, 10B active, NVFP4, 84 GB) | 500k, vLLM, stretched from 256k, 16-bit cache | 253k | 93k 3/3, 253k 3/3, 492k 2/3 | yes | 14 tok/s | 1,042 tok/s |
| 10 | Qwen3.8-Flash-Next (125B, 6B active, NVFP4, 133 GB) | 524k, stretched from 256k, FP8 cache; past 256k it returns the neighbouring record | 253k | 93k 3/3, 253k 3/3, 411k 2/3, 505k 2/3, 516k 2/3 | yes | 44 tok/s | 2,650 tok/s |
| 11 | Seed-OSS-36B (dense 36B, 8-bit GGUF, 38 GB) | 490k, llama.cpp, 8-bit cache | 253k | 93k 3/3, 253k 3/3, 457k 0/3 | yes | 6 tok/s | 114 tok/s |
| 12 | DeepSeek V4 Flash, 2-bit with DSpark (284B, 13B active, 2-bit GGUF and an 8-bit draft, 108 GB) | 131k, llama.cpp, 16-bit cache, reasoning effort max | 55k | 55k 3/3 | yes | 44 tok/s | 326 tok/s |
| 13 | Laguna S 2.1 (118B, 8B active, NVFP4, 100 GB) | 500k, trained to 1M, 500k fits here, FP8 cache; thinks on some requests only | none | 93k 2/3, 253k 2/3, 411k 1/3 | tool loop only | 12 tok/s | 1,520 tok/s |
| 14 | Nemotron 3 Super (120B, 12B active, NVFP4, 80 GB) | 1M, native window, FP8 cache | none | 93k 2/3, 235k 1/3, 489k 1/3 | yes | 16 tok/s | 1,045 tok/s |
| 15 | Nemotron 3.5 Lightning (30B, 3B active, NVFP4, 22 GB) | 1.05M, vLLM, 16-bit cache | none | 89k 2/3, 253k 1/3, 521k 1/3, 760k 1/3, 991k 1/3 | yes | 124 tok/s | 2,411 tok/s |
| 16 | Nemotron-Labs-3 Puzzle 75B (75B, 9B active, NVFP4, 54 GB) | 1M, vLLM, 16-bit cache | none | 89k 2/3, 253k 1/3, 521k 1/3, 760k 0/3, 950k 0/3 | yes | 20 tok/s | 1,565 tok/s |
| 17 | Nemotron-Cascade-2 30B-A3B (30B, 3B active, 8-bit GGUF, 34 GB) | 1.05M, llama.cpp, 16-bit cache | none | 89k 2/3, 253k 0/3, 521k 0/3, 760k 0/3, 991k 0/3 | yes | 58 tok/s | 1,821 tok/s |
| 18 | Kimi Linear 48B-A3B (48B, 3B active, 8-bit GGUF, 52 GB) | 1.05M, llama.cpp, 16-bit cache | none | 61k 0/3, 256k 2/3, 502k 0/3, 751k 0/3, 986k 0/3 | tool loop only | 58 tok/s | 1,112 tok/s |
| 19 | MiMo-V2.6-Flash, REAP 50% (half the experts kept, FP4 experts, 92 GB) | 1M, llama.cpp, 8-bit cache | none | 89k 0/3, 254k 0/3, 521k 0/3, 760k 0/3, 954k 0/3 | no | 14 tok/s | 590 tok/s |
<!-- results:end -->

How they fail matters as much as where. The Nemotron models return a code
from the wrong part of the prompt. Nemotron Puzzle 75B, which NVIDIA cut down
from Super and distilled at long context, mixes up record numbers that share
digits: it gave locker 2998's code for locker 3298 at 89k, locker 961's for
locker 9611 at 253k and locker 1917's for locker 19167 at 521k. At 760k it
went looking for locker 13966, found record 1396 and searched until its
32,768 tokens ran out. At 950k it wrote the first code and then the other
two locker numbers as if they were codes, and stopped without calling the
tool. Flash-Next past its trained length returns the neighbouring record:
asked for locker 9337 it gave locker 9338's code, and for locker 137 it gave
locker 37's, each time quoting the line it believed it had read. An agent
that is nearly right about what it read is worse than a slow one.

Every Qwen window past 256k is stretched. The models are trained to 262,144
tokens and their cards document YaRN for more, which rescales positions for
every prompt, long or short. What that does depends on the model, so it has
to be tested each time:

- Qwen3.8-27B, with dense attention in a quarter of its layers, stayed exact
  at 507k with a factor of 4 and a 16-bit cache, and at 519k with an FP8
  cache. At 986k with the FP8 cache it got two of three: asked for locker
  34548 it gave locker 34458's code, two digits swapped, after 46 minutes of
  reading.
- Qwen3.8-Flash-Next attends sparsely and loses its place past the trained
  length. A factor of 2 with a 524k window does better than a factor of 4 with
  558k (2/3 against 0/3 at about 517k). Neither is exact.
- Qwen3.6-35B-A3B, small and fast at 78 tok/s, stayed exact at 253k with a
  factor of 4, then got 2/3 at 519k, 760k and 960k when measured again. Each
  miss came from a record whose number looks like the one asked for: locker
  13328's code for 9338, 8536's for 13486 and 33646's for 33654. The first
  run got 1/3 at 1M.
- Qwen3.5-122B-A10B stayed exact at 253k stretched to 500k with a 16-bit
  cache, and at 519k stretched to 1.01M with an FP8 cache. Past that it came
  apart: one of three at 760k, and none at 960k, where even locker 137 came
  back wrong. With the 16-bit cache it gave locker 870's code for locker 8870
  at 492k.
- Qwen3-Coder-Next, an older model whose card only claims 256k, stopped
  calling tools altogether at a factor of 4. It is not in the table.

DeepSeek V4 Flash, GLM-5.3-Flash and Kimi Linear are trained to 1M, and most
of their layers keep a compressed or linear cache, so a full window costs a
few GiB. The first two only fit this box at 2 to 3 bits per weight, through
llama.cpp, which is a harsher cut than the 4-bit builds the rest run at.

DeepSeek's template asks it to reason at length only when the request sets a
reasoning effort of high or max, and the profiles set max. At max the 3-bit
build was exact at every size tried, up to 805k, and no answer ran past 600
tokens. Its reasoning opens by saying it will search for each record before
it names a code. At its default effort it was exact to 255k. At 502k it wrote
down the wrong code for the middle locker while thinking and never called the
tool. Given 32,768 tokens instead of 4,000, it went back and forth over that
code until they ran out. Two more tries at normal sampling got one and two of
three. At 970k it got two of three, giving locker 26499's code for locker
26995. The max run had an 857k window, less than the earlier ones because
downloads were holding memory. With the 10 GB margin every model now keeps,
the 3-bit build would get far less than that, so it is not run again near 1M.
The 2-bit build is.

The 2-bit build, also at max, had room for the full 1,048,576 tokens. It got
all three codes at 61k, 255k, 501k and 985k, and 985k is the largest prompt
any model here has answered exactly. At 751k it gave locker 20999's code for
locker 20909. It quoted record 20909 with that code, said it would check, and
quoted it the same way again. So it is exact to 501k by the ranking's rule,
and the 3-bit build stays ahead.

Pruned to 150B with DeepSeek's own FP4 experts, it also had room for the full
window, decoded at 16 tok/s and read 460 tok/s. It was exact to 501k and got
two of three at 751k and at 985k. Both wrong codes came from far off in the
prompt: locker 12503's for locker 20909, and locker 11427's for locker 27428.
Pruned to 200B at 3 bits, it decoded at 17 tok/s and got all three at 985k,
but one wrong at 501k and one at 751k, again from far off: locker 13412's
code for locker 13984, and locker 40795's for locker 20909. Neither pruned
build does as well as the whole model at 2 bits.

DSpark, DeepSeek's draft model, more than doubles decode speed: the 2-bit
build with it decoded at 44 tok/s against 20 without, and the main model
accepted 90 to 95 percent of the drafted tokens. The answers are the main
model's, and it got all three codes at 55k. The draft's 11 GB comes out of
the cache, though, and only a 131k window fit, so on this box the profile is
no use past about 90k.

MiMo-V2.6-Flash, cut to half its experts with FP4 experts, had room for its
full 1M window with 10 GB to spare and decoded at 14 tok/s. It got no code
right at any size. At 89k its reasoning found two of the three codes and
misread the first, and then it sent the three locker numbers as the codes.
At 254k it quoted all three records with the wrong codes. At 521k it began
listing a code for every record and ran out of its 32,768 tokens after two
hours. At 760k it wrote a string of made-up numbers, and at 954k it counted
1, 2, 3 for three hours. It called no tool at either size. On the short
check it answered without thinking or calling the tool. Half the experts is a
deep cut and nobody has published evaluations of this build, so this says
little about the whole model, which does not fit here.

Kimi Linear has no thinking mode. It had room for its whole 1M window with
57 GB to spare, decoded at 58 tok/s and read 1,100 tok/s, and its tool calls
were well formed at every size. At 61k, 502k, 751k and 986k it sent the three
locker numbers back as the codes, writing about 50 tokens each time. At 256k
it got the first and last codes and gave locker 2168's code for locker 7168.

Seed-OSS-36B is dense and trained to 512k. With an 8-bit cache and the 10 GB
margin it got a 490,240-token window, so its last step was 457k. It got all
three codes at 93k and 253k, after 1,015 and 482 tokens. At 457k it began the
same way, then wrote its answer out as text and stopped after 734 tokens
without calling the tool. The log kept only the start of that text, so
whether its codes were right is not known; the test now notes which codes
such an answer contains. Dense attention makes it slow at depth. It decodes
6.3 tok/s on a short prompt and 3.2 at 93k, reads about 650 tok/s at first
and under 40 past 400k, and its last step took nearly two hours.

Nemotron-Cascade-2 is Lightning's shape with different post-training, and
its card shows needle tests passing at 1M. It had room for the whole window
with 76 GB to spare, decoded at 58 tok/s and went through all five steps in
44 minutes, but recall broke early. At 89k it gave locker 1522's code for
locker 1762. At 253k all three were wrong, each from a different record:
locker 1373's code for 137, 899's for 4919 and 861's for 9611. At 521k and
760k it sent the locker numbers back as the codes. At 991k it began a tool
call with the three locker numbers and then repeated two codes until its
32,768 tokens ran out.

Lightning, measured again the same way with a 16-bit cache, did worse than
in its first run, which had an FP8 cache and got all three at 102k. This time
it got two of three at 89k, and from 253k to 991k only locker 137 came back
right. The wrong codes come from records whose numbers share most of their
digits with the one asked for: locker 919's code for 4919, 911's for 9611,
1167's for 19167 and 3595's for 35985.

Long prompts cost time. Flash-Next reads about 2,500 tok/s, so a cold 500k
prompt takes under four minutes. The dense 27B reads 900 tok/s for the first
100k and under 300 past 250k, and the same prompt takes 22 minutes. The
server caches what it has read, and a request that extends the previous
prompt only pays for the new part. For these hybrid models the cache resumes
from the end of an earlier prompt, so anything that changes the start of the
conversation, or shortens it, means reading it all again. The agent keeps its
system prompt fixed for that reason.

## Profiles

A profile in `config/vllm/` names the weights, the window, the parsers and the
memory share. `VLLM_PROFILE` and `VLLM_LONG` in `config/spark.env` pick one.

| profile | what it is | windows |
|---|---|---|
| `qwen3.8-27b` | dense 27B, exact past 500k, 16-bit cache | 256k native; `VLLM_LONG=1` or a token count stretches it to 900k |
| `qwen3.8-27b-fp8` | the same model with an FP8 cache, which makes room for a full 1.01M | 256k native, 1.01M stretched |
| `qwen3.8-flash-next` | fastest and strongest coder, served through a patched vLLM (below) | `VLLM_LONG=0` asks for 500k, `1` for the most that fits |
| `qwen3.6-35b-a3b` | 35B with 3B active, small and fast | 256k native, 1.01M stretched |
| `qwen3.5-122b`, `qwen3.5-122b-fp8` | 122B with 10B active, with a 16-bit or an FP8 cache | 256k native, 500k or 1.01M stretched |
| `deepseek-v4-flash-spark` | DeepSeek V4 Flash in 0xSero's single-Spark build: pruned experts, 3-bit EXL3, DSpark draft, through a patched vLLM (below) | 262k at most |
| `deepseek-v4-flash-ds4` | DeepSeek V4 Flash on ds4, an engine written for this one model, with antirez's 2-bit GGUF and the DSpark draft (below) | 1M |
| `laguna-s-2.1` | Poolside's 118B coding model, trained to 1M | what its 93 GiB of weights leave room for |
| `nemotron-3-super`, `nemotron-3.5-lightning` | native 1M windows that fail the recall test | 1M |
| `nemotron-puzzle-75b` | Super cut down to 75B with 9B active and distilled at long context | 1M (the card's; its config says 256k) |
| `gpt-oss-120b` | 128k | 128k |

llama.cpp profiles in `config/models/`:

| profile | what it is | windows |
|---|---|---|
| `deepseek-v4-flash` | DeepSeek V4 Flash, 284B with 13B active, 3-bit GGUF | trained to 1M; the largest that fits |
| `deepseek-v4-flash-q2`, `deepseek-v4-flash-dspark` | the same at 2 bits, alone and with DeepSeek's DSpark draft model for faster decoding | the largest that fits: 1M alone, 131k with the draft |
| `deepseek-v4-flash-reap150`, `deepseek-v4-flash-reap200` | the same with experts pruned to 150B (in DeepSeek's own FP4, 85 GB) or 200B (3-bit, 93 GB) | the largest that fits |
| `glm-5.3-flash` | GLM-5.3-Flash, 320B with 18B active, 2-bit GGUF | trained to 1M; the largest that fits |
| `glm-5.3-flash-reap-coder` | the same pruned to 144 experts and tuned on coding sessions, 4-bit | the largest that fits |
| `mimo-v2.6-flash-reap` | Xiaomi MiMo-V2.6-Flash with half its experts, FP4 experts | trained to 1M; the largest that fits |
| `dots3-note` | dots3-note preview, 280B with 16B active, 2-bit | trained to 512k; the largest that fits |
| `k2-horizon-mova` | K2-Horizon MoVA, 36B with 4B active, 8-bit weights and cache | trained to 512k; the largest that fits |
| `nemotron-cascade-2` | Nemotron-Cascade-2, 30B with 3B active, 8-bit | 1M; the largest that fits |
| `kimi-linear-48b` | Kimi Linear 48B with 3B active, 8-bit, no thinking mode | trained to 1M; the largest that fits |
| `seed-oss-36b` | Seed-OSS-36B, dense, 8-bit weights and cache | trained to 512k; the largest that fits |
| `clef`, `clef-flash` | Cloudflare's decision models: typed answers at `/v1/systemone`, no text | passages up to 64k |
| `gpt-oss-120b`, `gpt-oss-20b` | OpenAI's open models | 128k |

`CTX=fit` in a llama.cpp profile leaves the window to llama.cpp: it starts
from the window the model was trained for and shrinks it until weights and
cache fit, keeping `FIT_MARGIN_MB` free for the rest of the machine. That
defaults to `SAFETY_GB` plus 6 GB, 16 GB in all, because llama-server's own
memory comes on top of what the fit counts. The window it settled on is in
the server log and on the control page. It depends on what else holds memory
at the time: with the 8 GB margin used until 2026-10-15, the 3-bit DeepSeek
got 991k with nothing else running and 857k while a download held about
3 GB, and the 2-bit build, 7 GB smaller, got the full 1,048,576 with a
download running.

`TEMPLATE_KWARGS` in a llama.cpp profile passes values to the model's chat
template, for every request that does not set them itself. The DeepSeek
profiles set `reasoning_effort` to max and gpt-oss to high, the deepest each
template knows. Most other templates already default to their deepest
setting or only switch thinking on and off.

When vLLM refuses a window because the cache cannot hold it, it says what
would fit. The launchers take 97% of that figure, write it to
`run/vllm-ctx-<profile>` (`run/flash-next-ctx` for Flash-Next) and start again
with it, and later starts go straight there. Delete the file to have the full
window tried again.

Five ways to serve, one port:

- vLLM in NVIDIA's DGX Spark container (`spark/vllm.sh`). Proper tool-call
  and reasoning parsers, prefix caching, and an Anthropic-style
  `/v1/messages` endpoint for Claude Code. The default.
- Qwen3.8-Flash-Next through `spark/flash-next.sh`. Its checkpoint is larger
  than the machine's memory, so the stock image cannot load it. The launcher
  uses the [qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
  recipe (Apache-2.0), cloned under `third_party/` at a pinned commit, which
  patches vLLM 0.31 to read the 48 GiB n-gram table from the NVMe. It sets
  itself up on first use: the image build and checkpoint preparation take
  about twenty minutes once.
- DeepSeek V4 Flash through `spark/deepseek-spark.sh`, which runs
  [0xSero's build](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)
  for one Spark (Apache-2.0, weights MIT) from its image, pinned by digest.
  The image would download and repack the weights at its first start; here
  `vllm.sh pull` fetches them and the bake-off repacks them right before the
  model's turn, so a start only loads. A profile can name such steps with
  `PULL` and `PREPARE`. The image serves with thinking off unless a request
  asks for it; the script starts it with thinking on at max effort.
- DeepSeek V4 Flash through `spark/ds4.sh`, which builds
  [ds4](https://github.com/antirez/ds4) (MIT), antirez's C and CUDA engine
  for this one model, in [Entrpi's fork](https://github.com/Entrpi/ds4) for
  the GB10, cloned under `third_party/` at the tag Entrpi's installer pins.
  It runs only antirez's own GGUF builds. The 2-bit one is 81 GiB, and the
  DSpark draft beside it is used without being asked for. It is reported to
  read prompts at about 1,000 tok/s on one Spark, three times llama.cpp, and
  its cache holds a full 1M window in about 4 GiB. It takes no API key and
  has no `/health`. The vLLM profile runs it under `vllm.service` with
  thinking on at max effort and a memory floor of `SAFETY_GB`.
- llama.cpp built from source (`spark/serve.sh`, GGUF profiles in
  `config/models/`). This is how models whose release weights are larger
  than the machine run here at all, at 2 to 3 bits per weight. It reads each
  model's own chat template, so tool calls need no parser setting, and it
  can run a small sidekick model on a second port when the main profile
  leaves memory for one.

## Layout

    config/spark.env.example   hostname, backend, model, port, API key (copy to spark.env)
    config/vllm/*.env          vLLM profiles: HF repo, window, parsers, memory share
    config/models/*.env        llama.cpp profiles: GGUF repo, context, KV type
    spark/bootstrap.sh         one-shot setup on the Spark; idempotent; no sudo
    spark/build-llama.sh       llama.cpp with CUDA for sm_121, into llama.cpp/
    spark/pull-model.sh        GGUF download into models/<key>/ (checks the repo first)
    spark/vllm.sh              pull | serve | switch | install | stop | logs | image
    spark/flash-next.sh        Qwen3.8-Flash-Next through the pinned third-party recipe
    spark/deepseek-spark.sh    DeepSeek V4 Flash through 0xSero's single-Spark build
    spark/ds4.sh               DeepSeek V4 Flash on ds4: fetch | build | serve
    spark/serve.sh             llama-server for a profile (main, sidekick, or any key)
    spark/service.sh           systemd user units for the llama.cpp instances
    spark/headless.sh          (sudo) bring the Wi-Fi back when it waits for a password nobody types
    spark/wifiwatch.sh         the Wi-Fi check that headless.sh installs
    spark/panel.py             control page: status, profile and context switch, connection strings
    spark/preflight.sh         memory, GPU processes, ports, does the profile fit
    spark/bake.sh              measure a list of profiles in turn: download, serve, scripts/bench.py
    spark/bench.sh             llama-bench at 0 and 8k depth
    spark/ollama.sh            fallback: Ollama from the tarball, in-tree
    scripts/deploy.sh          rsync this tree to the Spark (and run bootstrap); leaves its spark.env alone
    scripts/tunnel.sh          ssh -L for the model ports and the control page
    scripts/status.sh          what the Spark is doing
    scripts/smoke-test.py      tool-calling check, standard library only
    scripts/context-test.py    recall across the window through a tool call; the acceptance test
    scripts/bench.py           speed, thinking, tool loop and the recall test, one JSON line per model
    scripts/results.py         rank the results; --publish writes docs/results.json and the tables
    scripts/clef-test.py       the recall test for a decision model: multiple choice and claims to check
    bin/frank                  frank in the current directory, pointed at the server; tmux over SSH
    bin/frank-start            pick one model (and the pages) to start, then bin/frank
    agent/                     frank, the agent (see agent/README.md)
    docs/                      the project page, served by GitHub Pages
    claude-code/               run Claude Code against the Spark
    Makefile                   the above as targets; make help
    LICENSE.md                 PolyForm Noncommercial 1.0.0

## First run

On the Mac:

    make config                       # writes config/spark.env with a fresh API key
    $EDITOR config/spark.env          # SPARK_HOST, SPARK_USER, BACKEND, VLLM_PROFILE or MODEL
    make deploy                       # rsync to ~/Development/frank on the Spark (or copy the tree however you like)

On the Spark, in the tree:

    spark/bootstrap.sh                # tools into .tools/, llama.cpp build, weights, services, smoke test
    spark/bootstrap.sh --no-pull      # same without the downloads

The bootstrap builds llama.cpp (about 15 minutes on the 20 cores), pulls the
container image (37 GB) and the weights of the profile in `config/spark.env`
(22 GB for Qwen3.8-27B, 133 GB for Flash-Next), installs systemd user units,
starts the model server, waits for the health check and runs the smoke test.
Nothing needs sudo; linger is already on, so the services outlive the SSH
session. Nothing starts at boot. The pages and the model server start when
someone asks for them (`spark/service.sh start`, the page, a bake) and never
restarts itself.
Rerunning it is safe. The downloads are the slow part: this Spark gets 20 to
30 MB/s from Hugging Face, so a 100 GB model takes an hour or more.

Back on the Mac:

    make panel                        # control page at http://omega.local:8088
    make chat                         # chat page at http://omega.local:8089
    make tunnel &                     # localhost:8080 (and :8081) -> Spark
    make agent-install
    make doctor                       # models, context window, tok/s, thinking and tool-call check
    make agent                        # interactive session in the current directory

Then the test that decides whether a profile is worth using, on the Spark so
it does not depend on the laptop staying awake:

    python3 scripts/context-test.py --base-url http://127.0.0.1:8080/v1 --api-key KEY --sizes 100,250,500

The agent runs anywhere with network access to the Spark, and does not need
this checkout there. The control page prints the command, which looks like
this:

    FRANK_BASE_URL=http://omega.local:8080/v1 \
    FRANK_API_KEY=... \
    uvx --from "frank-local @ http://omega.local:8088/dist/<hash>/frank_local-0.3.0.tar.gz" frank

To run it on the Spark itself: `spark/bootstrap.sh --agent`, then
`agent/.venv/bin/frank`.

## The frank command

`bin/frank` starts frank in the current directory, already pointed at the
model server. `spark/bootstrap.sh` links it into `~/.local/bin` on the Spark,
and `make frank-command` does the same on a Mac, where it talks to the Spark
over the network:

    ssh omega.local
    cd ~/Development/steel
    frank                             # or frank -c to resume the last session here

Nothing is loaded after the Spark starts. `frank-start` lists the downloaded
models with the memory each wants free, and the two pages. It starts the one
model you pick and the pages you pick, nothing else, waits until the model
answers and runs `frank` with whatever options it was given. Picking another
model stops the one that is running; the pick goes into `config/spark.env`, and
a model that leaves less than `SAFETY_GB` free once loaded is stopped again.
`--server` stops before the agent.

Run in your home folder, it lists `~/Development`, most recently changed
first, and starts in the one you pick. Paste a git URL at that prompt, or run
`frank <git url>` anywhere, and it clones the repository into `~/Development`
first (or opens the folder if it is already there). Over SSH it runs inside tmux, one
session per directory, so a dropped connection or a closed laptop leaves the
agent working; `frank` in the same folder attaches to it again, and Ctrl-b d
leaves it running on purpose. Quitting the agent leaves a shell in the
project. Any frank option passes through (`frank --effort max`, `frank
--doctor`). `FRANK_TMUX=0` turns tmux off.

On the Spark the agent works on the Spark's copy of a project, so a project
has to be cloned or synced there first. To make plain `ssh frank` land in
the list, add this to `~/.ssh/config` on the Mac:

    Host frank
        HostName omega.local
        RequestTTY yes
        ForwardAgent yes
        RemoteCommand bash -lc frank

ForwardAgent lets git inside the session push with the Mac's keys; the
session keeps a stable link to whichever connection is current.

## Control page

`spark/panel.py` is a small page served from the Spark, standard library only,
installed as the user unit `frank-panel` by the bootstrap. It is on the LAN at
http://omega.local:8088 (`make panel` opens it). There is no login.

It shows what is being served, the context window and memory, and has buttons
for start, stop and restart. Downloads in progress get a bar each, with speed
and a finish time. The profile table lists every vLLM and llama.cpp profile
with its window range, how it thinks, its tool parser and whether the weights
are on disk; pick one, pick a window and it rewrites `config/spark.env` on the
Spark, reinstalls the unit and restarts. Weights can be pulled from the same
table. The Reliability table ranks every model measured so far by how far it
stays exact, with the codes it got right at each prompt size and its decode and
read-in speed, and says which model a running bake-off is on. The Connect box prints the full command for Claude Code and for frank
against whichever host name you choose, with copy buttons, and two tests: one
request to `/v1/messages`, which is what Claude Code needs, and the
tool-calling check. The host list has every name the Spark answers to
(`omega.local` on the LAN, the bare name for Tailscale or a DNS entry of your
own, its addresses) and a box for any other. Both commands work on any
machine that can reach the Spark: the page hands frank out as an installable
archive, so its command is two variables and one `uvx` line. Output of whatever ran last is at the bottom, next to the
service logs. The page updates itself over server-sent events.

`PANEL_HOST=0.0.0.0` in `config/spark.env` is what puts it on the LAN. Set it
to `127.0.0.1` to keep the page to the SSH tunnel (http://localhost:8088 with
`make tunnel` running). The page shows the API key and can restart services,
and anyone on the network can open it.
A profile switched from the page changes `config/spark.env` on the Spark,
which is the copy that counts: `make deploy` sends the Mac's only when the
Spark has none. `make vllm-switch` changes both.

## Chat page

The `frank-chat` unit runs `frank --web`, which is frank in the browser at
http://omega.local:8089 (`make chat` opens it). It looks like ChatGPT or Claude
on the web, with the chats down the side, and each chat is a frank session in a
project folder on the Spark, with the same tools, thinking levels and permission
prompts as the terminal. A new chat asks which folder; the list is the folders
used lately and then the rest of `~/Development`.

The page and the `frank` command share the session files in `agent/.frank/`.
Terminal sessions show up on the page, `frank -c` in a project folder resumes
the chat last used there whichever side it was on, and Up in the message box
brings back what was typed in either. One side at a time can run a turn in a
session. The other says the session is busy until the turn is done, then picks
up what was added before it goes on. Each side builds its own system prompt, so
when a session moves from one to the other the model server may have to read
the whole conversation again.

Each chat runs its agent in a process of its own, so Stop works the way Ctrl-C
does, the model request or the shell command included. Slash commands work in
the message box too; `/clear` starts a new chat.

Like the control page it is started by hand, needs no model to be running, and
listens where `PANEL_HOST` says, so it is on the LAN with the shipped config.
There is no login. Anyone who can reach port 8089 can have the agent run
commands on the Spark as you. `PANEL_HOST=127.0.0.1` keeps both pages to the
SSH tunnel, which forwards 8089 as well. Elsewhere, `frank --web` serves the
same page for the sessions on that machine, on 127.0.0.1 unless `--host` says
otherwise.

## Day to day

    make status                       # memory, GPU, services, endpoints
    make logs                         # follow the main server
    make restart | stop | start
    make smoke                        # tool-calling check through the tunnel
    make switch MODEL=deepseek-v4-flash   # llama.cpp backend: change profile and restart
    make test                         # agent unit tests

vLLM profile changes go through one target, which edits `config/spark.env`
here and on the Spark, reinstalls the unit and restarts:

    make vllm-switch PROFILE=qwen3.8-27b LONG=1               # exact past 500k, slow
    make vllm-switch PROFILE=qwen3.8-27b LONG=0               # the same model at its native 256k
    make vllm-switch PROFILE=qwen3.8-flash-next LONG=0        # fast, exact inside 256k

Each profile is its own download; the bootstrap pulls only the one in
`config/spark.env`, so run `spark/vllm.sh pull <profile>` on the Spark, or use
the page, before switching to another. To try a profile in the foreground
first: `systemctl --user stop vllm; spark/vllm.sh serve qwen3.8-27b`.

Memory is checked before a model starts (`MIN_FREE_GB` in each profile), and
the start waits until the available memory has held steady for 30 seconds. A
server that was just stopped or killed takes a while to give its memory back,
and a model that began loading in that gap stalled the Spark for hours, three
times. `spark/preflight.sh` shows what is using the box if a start is
refused; `FORCE=1` skips the check and the wait.

Once loaded, a model must leave `SAFETY_GB` free, 10 GB unless
`config/spark.env` says otherwise. A switch from the page, `service.sh` or
`vllm.sh` stops a model that leaves less, and says so. While a bake measures
a model it also watches the memory and stops the server if it stays under
`GUARD_GB` (3 GB) for four seconds.

Other jobs on the Spark can take memory after a model has loaded. Some look at
what is free once, when they start, and allocate it minutes later after
compiling, so a model loaded in between gets caught short. That is what hung
the Spark under GLM-5.3-Flash: it was 61% into a 144k prompt, with free memory
flat at 2 GB, when another job's first training step took the rest. The bake
gives way to the jobs in `NEIGHBOURS` (`steel.service` and `flint-run.service`
unless set otherwise). While one of them is running it stops its server and
loads nothing. It also makes room five minutes (`LEAD_MIN`) before a timer in
`NEIGHBOUR_TIMERS` fires (`flint.timer`), and measures again later the model
it stopped.

Nothing here reboots the Spark when memory runs out; the kernel's OOM killer
handles it. A watchdog that rebooted the machine instead was taken out on
2026-10-18, after it rebooted it over a model that had run out of room next
to `steel`'s training. What keeps memory from running out is one model at a
time with `SAFETY_GB` to spare, and room for the neighbours.

Coming back from a restart is not the same as being reachable. The Spark is on
Wi-Fi, and when a Wi-Fi handshake fails NetworkManager takes the saved password
for wrong and asks for a new one. After the restart on 2026-10-16 the first
handshakes timed out, the login screen's password box took the request, and the
Spark stayed off the network for 16 hours until someone logged in. `sudo
spark/headless.sh install` adds a root timer that checks the Wi-Fi every
minute. When it has been waiting for a password, failed or disconnected on two
checks in a row, it connects it again, and NetworkManager uses the saved
password. A radio turned off with `nmcli radio wifi off` is left alone.
`spark/headless.sh status` shows it. A cable to the router would make all of
this unnecessary.

## The agent

`frank` is a Python package in `agent/`. It streams from any OpenAI-compatible
endpoint and gives the model: `bash` with a persistent working directory,
`read_file`, `write_file`, `edit_file` (exact-match replace with a diff),
`glob_files`, `grep_files`, `list_dir`, `web_fetch`, `todo_write` and
`spawn_subagent` (a fresh read-only agent whose report comes back as the tool
result, optionally on the sidekick model). Permissions ask before anything
that changes state, with an allowlist for obviously safe commands. The
context window is read from the server, tool output limits scale with it, and
the history is summarised when it fills. Sessions are saved and resumable.

Thinking has five levels (`--effort off|low|medium|high|max`, default `max`).
On a model that reasons natively the agent translates the level into whatever
that family calls it (Qwen3.8 says `xhigh`, some models only switch thinking
on or off) and keeps the reasoning in the history. On one that does not, the
system prompt asks for a `<think>` block before every answer and tool call,
and the levels change how much work that block has to do. Details in
`agent/README.md`.

The prompts are in `agent/frank_local/prompts/` as Markdown: the system
prompt (identity, how the session works, delivering work, working with code,
writing as the user, writing for the user, planning), the user's standing
instructions, the thinking protocol, the subagent brief, the compaction
brief, and the tool descriptions. A file named `<name>.local.md` replaces
`<name>.md` and is ignored by git, so `user.local.md` is the place for
instructions of your own that should travel with the agent. They are modeled on the instructions Claude Code runs under.

## Claude Code against the Spark

`claude-code/README.md`. Short version: `source claude-code/env.sh && claude`
when the server has `/v1/messages` (vLLM does), or run the LiteLLM proxy from
`claude-code/litellm.yaml` when it does not. The control page prints the same
variables as one command to copy. The servers default to each model's deepest
reasoning setting, so Claude Code gets it without asking; the README there
says where its own effort setting reaches the model.

## License

PolyForm Noncommercial License 1.0.0. The full terms are in `LICENSE.md` and at
<https://polyformproject.org/licenses/noncommercial/1.0.0>. Noncommercial use,
changes and redistribution are allowed; commercial use is not.

Required Notice: Copyright 2026 thingg.co

The license covers this repository. The models, llama.cpp, vLLM and the
container image it downloads onto the Spark are separate works under their own
licenses.

## Open ends

- A window of 1M that is also exact. DeepSeek V4 Flash at 3 bits and max
  effort is exact to 805k, the most its window held on that run. With the
  10 GB margin it would get far less, so it is not run again. At 2 bits it
  held the full 1,048,576 and got all three at 985k, but one wrong at 751k.
  It runs again with the margin, and on ds4 with antirez's 2-bit build.
  Qwen3.8-27B with an FP8 cache is exact at 519k and got two of three at
  986k. With a 16-bit cache that model passed at 507k; the cache that fits at
  that precision (950k tokens) has no headroom over a prompt that fills the
  window, so that profile stops at 900k, and its 900k step is queued.
- Running the bake-off: dots3-note and K2-Horizon MoVA are queued on the
  Spark. Then DeepSeek V4 Flash runs at 2
  bits again, to see whether it keeps its 1M window with the 10 GB margin,
  and on ds4. GLM-5.3-Flash follows, whole at 2 bits and pruned at 4 bits.
  The Spark hung during its 144k prompt, but another job's training caused
  that, not the model. After that, the models measured before run again with
  `scripts/bench.py`, Flash-Next included. 0xSero's single-Spark DeepSeek
  build is not in this round. Clef and Clef-Flash get `scripts/clef-test.py`, the same records as
  multiple choice and as claims to check, once their 64k profiles have been
  tried.
  Larger models from the same labs (DeepSeek V4.1 Flash at 510 GB, MiniMax
  M3, Kimi K3, GLM-5.3) have no build that fits one Spark. Inkling-Small
  (Thinking Machines) needs llama.cpp support that is still a draft.
- Flash-Next with a 16-bit cache. At that precision only about 290k fits, so
  the stretched results above are all FP8. Whether the cache or the sparse
  attention is what loses its place past 256k is not settled.
- A checker next to the main model. Cloudflare's Clef models answer typed
  questions (yes or no, pick one, a score) about a passage in one pass,
  without generating text, and llama.cpp serves them at `/v1/systemone`. The
  27B one is good at spotting claims a passage does not support, which is the
  failure that matters here. `scripts/clef-test.py` asks it to pick each code
  out of eight that include the neighbouring records' codes, and to judge
  true claims and off-by-one ones. The whole passage goes through in one
  batch, and the attention mask for a batch grows with its square, so the
  profiles stop at 64k.
- More than one answer per size for the leaders. The bake-off asks once at
  temperature 0. `context-test.py --samples K` asks again at the server's own
  sampling, which costs only the answer once the prompt is cached.
- Two slots on llama.cpp. With one, a subagent's request pushes the main
  conversation out of the cache and the next turn reads it all again, about
  40 minutes at 500k for DeepSeek. `-np 2 --kv-unified` keeps both; vLLM
  already takes four requests at once.
- The older llama.cpp profiles (`glm-4.5-air`, `gpt-oss-120b`, `qwen3-235b`)
  have not been run. `pull-model.sh` checks the repo before downloading and
  says which line to fix.
- Memory headroom. `MEM_UTIL` values are what worked on the night, with a
  desktop session and a download running. `spark/preflight.sh` and the
  control page show what is in use.
