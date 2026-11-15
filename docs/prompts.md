# Prompt templates (local LLM / VLM)

## Query rewrite / follow-up

Handled in `app/retrieval.py::rewrite_query` — concatenates prior user turn for short follow-ups and pronoun references.

## Image captioning (ingest)

From `app/index.py::caption_image_with_llava`:

> Describe this enterprise figure/diagram/chart in 2 detailed sentences. Mention numbers, entities, and trends where visible.

## Answer-time VLM

From `app/orchestrator.py::_vlm_interpret`:

> User question: {query}
> Inspect this document figure. Describe only what is visible that helps answer. Do not invent labels or numbers you cannot read.

## Grounded answer generation

System prompt in `app/orchestrator.py::SYSTEM_PROMPT_TEMPLATE` — evidence-only answers, citation tags, image path rules, insufficient-evidence refusal line, conflict disclosure, VLM interpretation marked `[VLM]`.
