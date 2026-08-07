# ADR 0012: Bind the canonical remote scheduler to DashScope

- Status: Accepted
- Date: 2026-08-08

## Context

The locked planning component originally posted to an OpenAI URL embedded in
upstream code. Replacing only the model name or credential variable would send
the new provider's key to the wrong service. The upstream request loop also had
no request timeout, retried malformed structured output without a finite
budget, and estimated cost using OpenAI-specific prices.

The canonical profile uses the remote model only to order degradation task
labels. Local perception and terminal scale generation remain local model
workloads; the scheduling request does not require image bytes.

## Decision

The production profile pins the scheduler to Alibaba Cloud Model Studio in the
Beijing region and to the dated `qwen3.7-flash-2026-07-15` snapshot. Provider,
official API base URL, region, model, credential variable, timeout, retry
budgets, JSON mode, disabled thinking, token budget, and temperature are one
validated configuration contract.

A small overlay-owned transport calls the provider's OpenAI-compatible Chat
Completions protocol directly. “OpenAI-compatible” describes the wire format;
the canonical endpoint, credential, traffic, and billing belong to DashScope.
The adapter rejects remote image inputs, non-official endpoints, non-`stop`
responses, model-identity drift, empty content, malformed response envelopes,
and invalid task-order JSON. Transport and structure retries are independently
bounded. Evidence records request IDs, token usage, response model, parameters,
and a hash of the endpoint host, but never credentials, prompts, or image data.

The upstream checkout stays unchanged. The overlay removes its provider-specific
cost estimate because an unverified price table is not runtime evidence.

## Consequences

- The canonical runtime does not call the OpenAI service or require an OpenAI
  account or API key.
- Changing provider, region, endpoint, or model invalidates the production
  preflight and requires an explicit compatibility update.
- Scheduler replacement can change restoration order, so earlier calibration
  and ablation evidence cannot be treated as equivalent.
- Task labels inferred from an image are still semantic data sent to the remote
  service and must be covered by dataset authorization and privacy review.

## Provider references

- [Model Studio endpoint and region table](https://help.aliyun.com/en/model-studio/base-url)
- [Qwen model and snapshot catalogue](https://help.aliyun.com/en/model-studio/text-generation-model/)
- [Structured JSON output contract](https://help.aliyun.com/zh/model-studio/qwen-structured-output)
