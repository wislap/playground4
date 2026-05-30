# Tool Rerank Baseline V2 Early Report

Date: 2026-05-26

This report records the early V2 stage of the N.E.K.O tool relevance reranking
baseline. The goal is not to present a finished model. The goal is to preserve
the experimental state clearly enough that later V3 work can be compared against
it without guessing what changed.

## 1. Problem Framing

N.E.K.O is not a pure task-execution agent. The primary scene is emotional
companionship with optional agent/tool help. This makes tool relevance different
from a normal assistant router:

- Topic similarity is not sufficient.
- A user mentioning a domain does not always authorize a tool call.
- The system must distinguish emotional support, passive context, ambiguous
intent, weak relevance, and explicit actionable requests.
- Agent tools are especially risky because they imply broader execution, while
plugin tools are often narrower.

The baseline therefore predicts a continuous confidence score for each
conversation-tool pair, then evaluates whether the highest ranked tools match
the calibrated labels inside each candidate set.

The current training target is the calibrated `confidence` field, not a binary
label. The confidence range in this dataset is approximately:

```text
min=-3.0902
p10=-1.2801
p50=-0.0011
p90= 1.2813
max= 3.0902
mean ~= 0
```

This shape is intentional: labels are continuous and approximately centered,
so the model can learn relative confidence rather than only positive/negative
classes.

## 2. Dataset State

Current dataset:

```text
data/runs/neko_v3_500
```

File counts:

```text
conversations.jsonl    500
candidate_sets.jsonl   500
judgments.jsonl        500
calibrated.jsonl      9500
```

Train/validation split:

```text
train conversation groups: 400
val conversation groups:   100
train pairs:              7600
val pairs:                1900
```

The split is group-based by `conversation_id`, so pairs from the same
conversation do not cross train/validation boundaries.

Tool universe:

```text
data/tool_universes/neko_real_v1              17 tools: 4 agents, 13 plugins
data/tool_universes/synthetic_plugins_v1      12 plugins
data/tool_universes/synthetic_plugins_v2      30 plugins
data/tool_universes/synthetic_plugins_v3      50 plugins
data/tool_universes/synthetic_plugins_v4      40 plugins

total: 149 unique tools
```

Each candidate set currently contains 19 tools. This matches the intended
construction pattern: all available agent tools plus a sampled slice of plugin
tools, with fixed tool ids and randomized candidate order.

Conversation scenario distribution:

```text
companion_chat_no_tool             79
assistant_suggested_tool_no_auth   52
boundary_or_refusal_no_tool        45
ambiguous_need_clarification       45
explicit_plugin_action             39
screen_context_weak_tool           38
emotional_support_no_tool          37
memory_recall_no_tool              37
proactive_context_weak_tool        34
agentic_task                       34
open_thread_followup_no_tool       33
passive_event                      27
```

Relevance mode distribution:

```text
no_tool_or_low_relevance          283
weak_or_requires_confirmation     117
actionable_tool_relevance         100
```

This is a useful early balance for N.E.K.O: most samples should not strongly
trigger a tool, but enough positive and weak cases exist to train ranking.

## 3. Data Loader And Pair Construction

The loader lives in:

```text
experiments/tool_rerank_baseline/data.py
```

For each row in `calibrated.jsonl`, it joins:

- calibrated pair label
- candidate set
- conversation
- tool definition from the merged tool universe

The pair object contains:

```text
sample_id
conversation_id
tool_id
label
raw_score
conversation_text
tool_text
scenario_type
relevance_mode
```

Conversation rendering includes:

- trigger
- scenario/auth/actionability context
- role-tagged message history
- attachment markers when present

Tool rendering currently includes:

- `tool_id`
- `kind`
- optional metadata name
- full `source_text`

Important limitation: tool fields are currently flattened into one text block.
The model does not yet know which tokens came from identity, description,
capabilities, examples, schema, or non-trigger guidance. This limitation matters
for V3.

## 4. Evaluation Metrics

The metric implementation is in:

```text
experiments/tool_rerank_baseline/metrics.py
```

Current metrics:

- `mae`: absolute error on continuous confidence.
- `mse`: squared error on continuous confidence.
- `spearman`: global rank correlation over all validation pairs.
- `top1_match`: whether the highest predicted tool is the highest labeled tool
  within the conversation candidate set.
- `topk_recall`: whether the highest labeled tool appears in predicted top-k.
  Current k is 3.
- `no_tool_fp_rate`: for groups where all labels are below the high-label
  threshold, whether the model still emits a high-confidence prediction.
- `high_conf_precision`: among pairs predicted above the high-confidence
  threshold, fraction whose label is also high.

Current high-confidence threshold:

```text
high_label_threshold = 1.0
high_pred_threshold  = 1.0
```

The current aggregate score used for checkpoint selection is:

```text
topk_recall - no_tool_fp_rate - mae
```

This is crude but useful in early research: it rewards ranking, penalizes false
tool activation on no-tool groups, and keeps continuous confidence calibration
from drifting too far.

## 5. Architecture Progression

### 5.1 V1: TF-IDF/SVD Pair MLP

Config:

```text
experiments/tool_rerank_baseline/config.toml
```

Architecture:

```text
conversation text -> TF-IDF -> SVD dense vector
tool text         -> TF-IDF -> SVD dense vector

features = [
  conversation_vector,
  tool_vector,
  abs(conversation_vector - tool_vector),
  conversation_vector * tool_vector
]

features -> MLP -> confidence
```

This version exists as the no-download fallback and sanity-check baseline.

### 5.2 V2: Frozen Jina Sequence Late Interaction

Historical config:

```text
removed single-split V2 sequence config
```

Encoder:

```text
jinaai/jina-embeddings-v5-text-nano-retrieval
```

Unlike a single pooled embedding, V2 caches token-level hidden states:

```text
conversation text -> [seq_len, hidden]
tool text         -> [seq_len, hidden]
```

Then the trainable head computes:

```text
conv_pool
tool_pool
abs(conv_pool - tool_pool)
conv_pool * tool_pool
conv_to_tool mean/max/topk similarity
tool_to_conv mean/max/topk similarity
```

These features go into an MLP regressor.

This is better than fixed pooled embeddings, but it still compresses the
interaction into a few summary features. It can tell that two texts are related,
but it cannot fully model why a tool should or should not be called in a
companionship-heavy conversation.

### 5.3 V2.1: Jina Sequence + Lexical Side Channel

Config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_v21.toml
```

V2.1 keeps the Jina sequence late-interaction path and adds 8 lexical features:

```text
word ngram TF-IDF row cosine
char ngram TF-IDF row cosine
BM25 score
conversation/tool token overlap ratios
tool_id token overlap ratio
binary tool_id token match
```

These features are concatenated to the semantic interaction features before the
MLP head.

The lexical side channel is not a replacement for embedding. It is an extra
evidence path. It helps because tool ids, capability names, product names,
domains, and short literal triggers carry useful signal that can be weakened by
semantic pooling.

### 5.4 V2.2: Gated Lexical Add

Config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22.toml
```

Fusion:

```text
semantic_features + gate * lexical_projection(lexical_features)
```

Dropout:

```text
lexical.dropout = 0.20
```

This version tested whether lexical evidence should be projected back into the
semantic feature space. It was too conservative and lost ranking quality.

### 5.5 V2.2b: Lighter Gated Lexical Add

Historical config:

```text
removed single-split V2.2b gated lexical config
```

Same fusion as V2.2, lighter lexical dropout:

```text
lexical.dropout = 0.05
```

This is the strongest early compromise so far. It preserves V2.1 top-k ranking
while slightly improving MAE and high-confidence precision.

### 5.6 V2.3: Gated Lexical Concat

Config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_concat_v23.toml
```

Fusion:

```text
concat(semantic_features, gate * lexical_features)
```

This keeps lexical evidence explicit while learning a per-feature gate. It
improved Spearman slightly, but top1 ranking dropped compared with V2.1/V2.2b.

## 6. Caching And Runtime

Jina sequence encoding is frozen and cached under:

```text
experiments/tool_rerank_baseline/cache
```

Cache keys include:

- split name
- model name
- max length
- exact input text list

The important behavior is lazy loading: when all sequence cache files hit, the
Jina tokenizer/model are not loaded. This makes repeated training experiments
fast enough to iterate on CPU.

Current cached sequence groups:

```text
train_conversations
train_tools
val_conversations
val_tools
```

This is one reason V2.x can support many head/fusion ablations cheaply.

## 7. Results

Validation set:

```text
100 conversation groups
1900 conversation-tool pairs
```

Best checkpoint per run:

| Run | Best Epoch | Top1 | Top3 | MAE | MSE | Spearman | High-Conf Precision | No-Tool FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF pair MLP V1 | 24 | 0.24 | 0.41 | 0.6974 | 0.9210 | 0.4953 | 0.5363 | 0.0000 |
| Jina sequence V2 | 9 | 0.52 | 0.60 | 0.5918 | 0.6644 | 0.6418 | 0.7472 | 0.0000 |
| Jina + lexical concat V2.1 | 12 | 0.75 | 0.89 | 0.5548 | 0.5653 | 0.6879 | 0.7950 | 0.0000 |
| Jina + lexical concat V2.1 linear head | 12 | 0.26 | 0.36 | 0.6378 | 0.7030 | 0.6065 | 0.6857 | 0.0000 |
| Jina + gated add V2.2 | 6 | 0.66 | 0.79 | 0.5523 | 0.5470 | 0.6880 | 0.8812 | 0.0000 |
| Jina + gated add V2.2b | 6 | 0.75 | 0.89 | 0.5492 | 0.5494 | 0.6882 | 0.8704 | 0.0000 |
| Jina + gated add V2.2b linear head | 10 | 0.59 | 0.76 | 0.5703 | 0.5751 | 0.6774 | 0.7910 | 0.0000 |
| Jina + gated concat V2.3 | 10 | 0.68 | 0.88 | 0.5547 | 0.5564 | 0.6927 | 0.8662 | 0.0000 |

Main observations:

- V2 is a substantial improvement over TF-IDF/SVD, confirming that token-level
  Jina sequence information is useful.
- Lexical features produce the largest single jump in ranking quality:
  top1 improves from 0.52 to 0.75 and top3 from 0.60 to 0.89.
- Direct lexical concat is strong but less precise on high-confidence outputs
  than the gated versions.
- Heavy gated-add dropout is too restrictive.
- V2.2b is the current best early model because it keeps V2.1 top1/top3 while
  improving MAE and high-confidence precision.
- Linear-head ablations are substantially weaker. This supports keeping the MLP
  head in the V2 baseline and suggests that the current semantic/lexical
  features require nonlinear combination.
- V2.3 has the best Spearman, but its top1 drop makes it less attractive as the
  default router baseline.

Current recommended CV baseline:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv.toml
```

## 8. Failure Modes

Worst validation errors from the strongest early model, V2.2b:

```text
conv_000349 plugin.wheelchair_route_rehearsal label= 2.324 pred=-1.256 raw=24
conv_000091 plugin.xiaohongshu_note_helper    label=-3.090 pred=-0.088 raw=1
conv_000091 plugin.ar_collectible_overlay     label=-3.090 pred=-0.147 raw=1
conv_000463 plugin.game_agent_minecraft       label=-2.699 pred=-0.115 raw=0
conv_000331 plugin.anki_deck_builder          label= 2.309 pred=-0.225 raw=32
conv_000462 agent.computer_use                label=-1.961 pred= 0.525 raw=0
conv_000140 agent.computer_use                label=-1.653 pred= 0.820 raw=0
```

These errors suggest several distinct problems:

1. Some true positives are missed when the lexical path does not expose enough
   direct evidence or when the positive signal is distributed across the whole
   context.
2. Some strong negatives remain too close to neutral when they share surface
   topic with the conversation.
3. Broad agents such as `agent.computer_use` are particularly prone to weak
   false-positive behavior because their descriptions are semantically general.
4. The current model can learn relevance, but not yet the deeper trigger logic:
   whether a related tool is actually appropriate inside an emotional/supportive
   conversation.

The recurring issue is not simply lack of lexical features. It is that the
semantic interaction is still shallow. The model sees token sequences, but the
trainable head compresses them into pooled vectors and a small number of
similarity statistics.

## 9. Interpretation

The early V2 experiments support three claims.

First, a frozen Jina sequence encoder is a good base for this task. It gives a
large gain over TF-IDF/SVD while keeping iteration cheap through caching.

Second, lexical evidence is highly valuable. In a tool router, exact phrases,
tool ids, capability names, product names, and domain terms carry real routing
signal. Treating lexical features as an additional information path is better
than viewing them as a legacy alternative to embedding.

Third, current V2 semantics are still too compressed. The model currently uses
several high-level summaries of the interaction. That is enough to produce a
usable baseline, but it is not enough to robustly distinguish:

```text
topic-related conversation
vs
authorized/actionable tool need
```

This is especially important for N.E.K.O because companionship conversations
often contain concrete domains without requiring tool execution.

## 10. Needed Ablations

Before moving too far into V3, these ablations should be run:

### 10.1 MLP Head vs Linear Head

Question:

```text
Are the current feature interactions genuinely nonlinear?
```

This ablation has now been run for the two most important lexical variants:

| Setup | Head | Top1 | Top3 | MAE | Spearman | High-Conf Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V2.1 lexical concat | MLP | 0.75 | 0.89 | 0.5548 | 0.6879 | 0.7950 |
| V2.1 lexical concat | Linear | 0.26 | 0.36 | 0.6378 | 0.6065 | 0.6857 |
| V2.2b gated lexical | MLP | 0.75 | 0.89 | 0.5492 | 0.6882 | 0.8704 |
| V2.2b gated lexical | Linear | 0.59 | 0.76 | 0.5703 | 0.6774 | 0.7910 |

Interpretation:

- The concat variant collapses under a linear head, especially on group ranking.
- The gated variant remains more usable with a linear head, probably because the
  gate itself is nonlinear and already reshapes lexical evidence.
- Even with the gate, linear is still materially below MLP.
- The V2 baseline should keep the MLP head. The feature set is not linearly
  sufficient for this task.

### 10.2 Lexical Side Channel Ablation

Already partially done:

```text
V2 no lexical        top1=0.52 top3=0.60
V2.1 lexical concat  top1=0.75 top3=0.89
```

This is strong evidence that lexical side-channel features matter.

### 10.3 Gating And Dropout Ablation

Current evidence:

```text
concat             strong ranking, weaker high-conf precision
gated_add 0.20     too conservative
gated_add 0.05     best current compromise
gated_concat 0.15  better Spearman, weaker top1
```

Further tuning is possible, but lower priority than deeper semantic structure.

## 11. V3 Direction

The next major model direction should not be more handcrafted summary features.
It should expose deeper semantic structure while keeping the frozen Jina cache
advantage.

Recommended V3 direction:

```text
conversation sequence
tool.identity sequence
tool.description sequence
tool.capabilities sequence
tool.examples sequence
tool.schema sequence
optional non-trigger / avoid sequence

field-level interaction
small attention/transformer or field-aware MLP head
lexical side-channel retained as auxiliary evidence
```

Why field-level embedding matters:

- Tool description, capability list, examples, and schema do not mean the same
  thing.
- Trigger examples may define applicability better than generic descriptions.
- Schema can reveal whether the user supplied enough actionable information.
- Tool identity/name can be useful but should not dominate broad semantic
  relevance.
- Non-trigger examples, once generated, can directly encode the boundary between
  companionship and tool activation.

The first V3 implementation should keep Jina frozen and cached. The trainable
part can remain small.

## 12. Current Recommendation

This report is a historical snapshot from the single-split phase. That training
mode is now deprecated; current model training must use grouped 5-fold CV.

```bash
uv run python experiments/tool_rerank_baseline/cross_validate.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv.toml
```

Immediate research tasks should be evaluated through the CV entrypoint:

1. MLP vs linear head ablation on V2.1 and V2.2b.
2. V3 field-level tool encoding prototype.

Do not treat V2.2b as final. Treat it as the first usable reference point.
