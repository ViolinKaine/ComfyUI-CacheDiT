# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ComfyUI custom node pack that speeds up DiT sampling by caching transformer outputs between steps, built on the `cache-dit` library (installed: 1.5.3 in `/mnt/ComfyUI_Main/.venv`).

## Repo / fork setup

- `origin` = `zhzLuke96/ComfyUI-CacheDiT` (fork that added Krea2 + `cache_mode` lightweight option). Upstream = `Jasonzzt/ComfyUI-CacheDiT`; Krea2 discussion is upstream issue #42 (fork author: Krea2 residual diff swings with LoRAs, adaptive cache rarely hits; lightweight mode ~1.6x but blotchy artifacts).
- Parent repo rules in `/mnt/ComfyUI_Main/AGENTS.md` apply (small direct fixes, no new deps, short comments).

## Commands

- No tests, no build. Lint only: `/mnt/ComfyUI_Main/.venv/bin/ruff check .`
- Syntax check without ComfyUI: `/mnt/ComfyUI_Main/.venv/bin/python -m py_compile nodes.py utils.py nodes_ltx2.py nodes_wan.py`
- Imports `comfy.*`, so real testing happens inside ComfyUI (the user runs it).

## Architecture

Three independent node modules merged in `__init__.py`:

- `nodes.py` - `CacheDiT_Model_Optimizer` ("⚡ CacheDiT Accelerator") for image DiTs (Z-Image/NextDiT, Qwen-Image, Flux/Flux2, Krea2/SingleStreamDiT, Custom).
- `nodes_ltx2.py` - LTX-2 optimizer (own global `_ltx2_cache_state`, video+audio paths).
- `nodes_wan.py` - WAN2.2 MoE optimizer (per-transformer state, `_original_forward_wan`).
- `utils.py` - `MODEL_PRESETS` (per-model fn/bn blocks, threshold, pattern), `_manual_extract_blocks` (finds `.layers` / `double_blocks`+`single_blocks` / `.blocks` / `.transformer_blocks`), cache-dit config builders, summary dashboard.

Flow in `nodes.py`:

1. `optimize()` clones the ModelPatcher, stores a `CacheDiTConfig` in `model_options["transformer_options"]["cache_dit_turbo"]` and a dict copy on `diffusion_model._cache_dit_config`, and registers `OUTER_SAMPLE` + `DIFFUSION_MODEL` wrappers keyed `cache_dit_turbo`. Model type "Auto" is resolved by transformer class name (duplicated in two places in `optimize()`; keep both in sync).
2. `_cache_dit_outer_sample_wrapper` reads step count from `sigmas`, then calls `_enable_cache_dit` on first run (or transformer change) and `_refresh_cache_dit` every run.
3. `_enable_cache_dit` picks one of two backends:
   - **Lightweight** (`_enable_lightweight_cache`): replaces `transformer.forward` with `cached_forward`, which skips the *whole* forward on a fixed warmup/skip_interval schedule and returns a cached result. Used when `cache_mode="lightweight"`, for class names `NextDiT`/`QwenImage`/`Flux`, or as fallback when BlockAdapter fails. State is the module-global `_lightweight_cache_state`, keyed by run context (prompt/seed/latent hash via `_context_key`) and result shape.
   - **cache-dit BlockAdapter** (everything else, incl. Krea2 in `auto` mode): wraps `transformer.blocks` in cache-dit `CachedBlocks_*` with adaptive residual-diff threshold. cache-dit patches `self.blocks` (via `unittest.mock.patch.object`) only for the duration of `transformer.forward`.
4. `disable()` / `enable=False` restores `_original_forward`, calls `cache_dit.disable_cache`, resets global state.

## Gotchas

- During a BlockAdapter-cached forward, `transformer.blocks` is the cache-dit wrapper, not the real blocks. Any other DIFFUSION_MODEL wrapper that reimplements the forward and loops `model.blocks` (e.g. `Krea2-Training-Free-Improvements` `krea2_guided_forward`, calling `block.mod`) crashes with `'CachedBlocks_Pattern_3_4_5' object has no attribute 'mod'`.
- Lightweight mode replaces `transformer.forward`, so it skips every DIFFUSION_MODEL wrapper inside it on cached steps too (other nodes' guidance never runs on those steps).
- `nodes.py` and `nodes_ltx2.py` both use the `_original_forward` attribute name on the transformer.
- Turbo models (Krea2 8 steps): warmup eats most steps; README says <6 steps is not worth it.
