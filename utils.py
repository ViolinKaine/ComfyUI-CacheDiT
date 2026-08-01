"""ComfyUI-CacheDiT: Utility Functions
=====================================

This module provides:
- Model preset configurations
- BlockAdapter construction
- Cache configuration builders
- Summary statistics formatting (ASCII dashboard)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ComfyUI-CacheDiT")


# =============================================================================
# Model Presets - Hardcoded Recommended Configurations (2026 Models)
# =============================================================================

@dataclass
class ModelPreset:
    """Preset configuration for a specific model type."""
    name: str
    description: str
    description_cn: str
    # Forward pattern
    forward_pattern: str
    # DBCache config
    fn_blocks: int  # Fn_compute_blocks
    bn_blocks: int  # Bn_compute_blocks
    threshold: float  # residual_diff_threshold
    max_warmup_steps: int
    # CFG settings
    enable_separate_cfg: Optional[bool]
    cfg_compute_first: bool = False
    # Advanced settings
    skip_interval: int = 0  # Force compute every N steps (0=disabled)
    noise_scale: float = 0.0  # Noise injection scale
    # Strategy
    default_strategy: str = "adaptive"
    # TaylorSeer
    taylor_order: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "forward_pattern": self.forward_pattern,
            "fn_blocks": self.fn_blocks,
            "bn_blocks": self.bn_blocks,
            "threshold": self.threshold,
            "max_warmup_steps": self.max_warmup_steps,
            "enable_separate_cfg": self.enable_separate_cfg,
            "cfg_compute_first": self.cfg_compute_first,
            "skip_interval": self.skip_interval,
            "noise_scale": self.noise_scale,
            "default_strategy": self.default_strategy,
            "taylor_order": self.taylor_order,
        }


# Hardcoded presets for 2026 mainstream models
MODEL_PRESETS: Dict[str, ModelPreset] = {
    # =========================================================================
    # Z-Image Series
    # =========================================================================
    "Z-Image": ModelPreset(
        name="Z-Image",
        description="Z-Image standard (recommended: 50 steps, cfg=4.0)",
        description_cn="Z-Image 标准版 (推荐: 50步, cfg=4.0)",
        forward_pattern="Pattern_1",
        fn_blocks=8,  # F8B0 as specified
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=25,  # Half of 50 steps for quality
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,  # No noise injection for quality preservation
        default_strategy="adaptive",
        taylor_order=1,
    ),
    "Z-Image-Turbo": ModelPreset(
        name="Z-Image-Turbo",
        description="Z-Image Turbo (distilled, 4-9 steps)",
        description_cn="Z-Image Turbo (蒸馏版, 4-9步)",
        forward_pattern="Pattern_1",
        fn_blocks=4,
        bn_blocks=0,
        threshold=0.15,
        max_warmup_steps=3,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.002,
        default_strategy="static",
        taylor_order=0,  # Disabled for low-step models
    ),
    
    # =========================================================================
    # Qwen-Image Series
    # =========================================================================
    "Qwen-Image": ModelPreset(
        name="Qwen-Image",
        description="Qwen-Image standard (2511/2512 edit)",
        description_cn="Qwen-Image 标准版 (2511/2512 编辑)",
        forward_pattern="Pattern_1",
        fn_blocks=1,  # F1B0 as specified
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=8,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
    
    # =========================================================================
    # Flux Series (supports Flux.1 and Flux.2)
    # =========================================================================
    "Flux": ModelPreset(
        name="Flux",
        description="Flux (balanced caching, supports Flux.1 and Flux.2)",
        description_cn="Flux (平衡缓存，支持 Flux.1 和 Flux.2)",
        forward_pattern="Pattern_0",
        fn_blocks=8,
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=4,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
    
    # =========================================================================
    # LTX-2 Video Series
    # =========================================================================
    # LTX-2 is an Audio-Visual Transformer that processes dual latent paths:
    #   - Video latents (hidden_states)
    #   - Audio latents (audio_hidden_states)
    # 
    # Block Architecture:
    #   Input:  (hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states)
    #   Output: (hidden_states, audio_hidden_states)
    #
    # cache-dit Integration:
    #   - Uses Pattern_1 which expects: In=(h,enc_h), Out=(enc_h,h)
    #   - LTX-2's audio_hidden_states is mapped to Pattern_1's "second output" (enc_h position)
    #   - This creates a semantic mismatch: cache-dit treats audio_h as encoder_h
    #   - Official cache-dit uses functor_ltx2.py to patch transformer.forward and reorder block args
    #   - ComfyUI lightweight cache bypasses this by caching full transformer output
    #
    # Pipeline Types:
    #   - T2V: Text-to-Video (diffusers.LTX2Pipeline)
    #   - I2V: Image-to-Video (diffusers.LTX2ImageToVideoPipeline)
    #   - Official serving uses CACHE_DIT_LTX2_PIPELINE env var to switch pipelines
    # "LTX-2-T2V": ModelPreset(
    #     name="LTX-2-T2V",
    #     description="LTX-2 Text-to-Video (temporal consistency)",
    #     description_cn="LTX-2 文生视频 (时序一致性优化)",
    #     forward_pattern="Pattern_1",
    #     fn_blocks=4,  # F4B4 for video
    #     bn_blocks=4,
    #     threshold=0.08,
    #     max_warmup_steps=6,
    #     enable_separate_cfg=False,
    #     cfg_compute_first=False,
    #     skip_interval=3,  # Force compute every 3 steps for temporal consistency
    #     noise_scale=0.001,
    #     default_strategy="dynamic",
    #     taylor_order=1,
    # ),
    # "LTX-2-I2V": ModelPreset(
    #     name="LTX-2-I2V",
    #     description="LTX-2 Image-to-Video",
    #     description_cn="LTX-2 图生视频",
    #     forward_pattern="Pattern_1",
    #     fn_blocks=4,
    #     bn_blocks=4,
    #     threshold=0.08,
    #     max_warmup_steps=6,
    #     enable_separate_cfg=False,
    #     cfg_compute_first=False,
    #     skip_interval=3,
    #     noise_scale=0.001,
    #     default_strategy="dynamic",
    #     taylor_order=1,
    # ),
    
    # =========================================================================
    # Krea2
    # =========================================================================
    "Krea2": ModelPreset(
        name="Krea2",
        description="Krea2 SingleStreamDiT (turbo, cfg=1)",
        description_cn="Krea2 单流 DiT (turbo, cfg=1)",
        forward_pattern="Pattern_3",
        fn_blocks=8,
        bn_blocks=0,
        # Measured Krea2 residuals were ~0.29-0.39; 0.12 never caches.
        threshold=0.35,
        max_warmup_steps=3,
        enable_separate_cfg=False,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=0,
    ),

    # =========================================================================
    # Custom / Fallback
    # =========================================================================
    "Custom": ModelPreset(
        name="Custom",
        description="Custom model (manual configuration)",
        description_cn="自定义模型 (手动配置)",
        forward_pattern="Pattern_1",
        fn_blocks=8,
        bn_blocks=0,
        threshold=0.12,
        max_warmup_steps=8,
        enable_separate_cfg=None,
        cfg_compute_first=False,
        skip_interval=0,
        noise_scale=0.0,
        default_strategy="adaptive",
        taylor_order=1,
    ),
}


def get_preset(model_type: str) -> ModelPreset:
    """Get preset configuration for a model type."""
    return MODEL_PRESETS.get(model_type, MODEL_PRESETS["Custom"])


def get_all_preset_names() -> List[str]:
    """Get list of all available preset names."""
    return list(MODEL_PRESETS.keys())


# =============================================================================
# Forward Pattern Utilities
# =============================================================================

PATTERN_DESCRIPTIONS = {
    "Pattern_0": "Return_H_First=True, In=(h,enc_h), Out=(h,enc_h) - Flux style",
    "Pattern_1": "Return_H_First=False, In=(h,enc_h), Out=(enc_h,h) - Qwen/LTX/Z-Image\n"
                 "               LTX-2: audio_h mapped as 'second output' in Pattern_1 abstraction",
    "Pattern_2": "Return_H_Only=True, In=(h,enc_h), Out=(h,) - Single output",
    "Pattern_3": "Forward_H_only=True, In=(h,), Out=(h,) - Hunyuan/Wan",
    "Pattern_4": "Return_H_First=True, In=(h,), Out=(h,enc_h) - Special",
    "Pattern_5": "Return_H_First=False, In=(h,), Out=(enc_h,h) - Special",
}


def get_forward_pattern(pattern_name: str):
    """Get ForwardPattern enum from cache_dit."""
    try:
        import cache_dit
        pattern_map = {
            "Pattern_0": cache_dit.ForwardPattern.Pattern_0,
            "Pattern_1": cache_dit.ForwardPattern.Pattern_1,
            "Pattern_2": cache_dit.ForwardPattern.Pattern_2,
            "Pattern_3": cache_dit.ForwardPattern.Pattern_3,
            "Pattern_4": cache_dit.ForwardPattern.Pattern_4,
            "Pattern_5": cache_dit.ForwardPattern.Pattern_5,
        }
        return pattern_map.get(pattern_name, cache_dit.ForwardPattern.Pattern_1)
    except ImportError:
        raise ImportError(
            "cache_dit library not found. Please install: pip install cache-dit>=1.2.0"
        )


# =============================================================================
# Cache Configuration Builders
# =============================================================================

def build_cache_config(
    num_inference_steps: Optional[int],
    fn_blocks: int,
    bn_blocks: int,
    threshold: float,
    max_warmup_steps: int,
    enable_separate_cfg: Optional[bool],
    cfg_compute_first: bool,
    skip_interval: int,
    strategy: str,
    scm_policy: Optional[str] = None,
):
    """
    Build DBCacheConfig with advanced settings.
    
    Args:
        num_inference_steps: Total inference steps (None for unknown)
        fn_blocks: Fn_compute_blocks
        bn_blocks: Bn_compute_blocks  
        threshold: residual_diff_threshold
        max_warmup_steps: Steps before caching starts
        enable_separate_cfg: CFG separation mode
        cfg_compute_first: Compute CFG first
        skip_interval: Force compute every N steps (0=disabled)
        strategy: 'adaptive', 'static', or 'dynamic'
        scm_policy: Steps computation mask policy
    """
    try:
        import cache_dit
        from cache_dit import DBCacheConfig, steps_mask
        
        config = DBCacheConfig(
            Fn_compute_blocks=fn_blocks,
            Bn_compute_blocks=bn_blocks,
            residual_diff_threshold=threshold,
            max_warmup_steps=max_warmup_steps,
            num_inference_steps=num_inference_steps,
        )
        
        # CFG settings
        if enable_separate_cfg is not None:
            config.enable_separate_cfg = enable_separate_cfg
        config.cfg_compute_first = cfg_compute_first
        
        # Strategy-based configuration with max_cached_steps
        if strategy == "static":
            # Static: More aggressive caching, fixed cache budget
            config.max_cached_steps = int(num_inference_steps * 0.5) if num_inference_steps else -1
            config.max_continuous_cached_steps = -1
        elif strategy == "dynamic":
            # Dynamic: Conservative caching with limits
            config.max_cached_steps = int(num_inference_steps * 0.7) if num_inference_steps else -1
            config.max_continuous_cached_steps = 4  # Limit continuous caching
        else:  # adaptive (default)
            # Adaptive: Unlimited caching based on threshold
            config.max_cached_steps = -1
            config.max_continuous_cached_steps = -1
        
        # Apply SCM policy or skip_interval
        if num_inference_steps is not None:
            if scm_policy and scm_policy != "none":
                # Use predefined SCM policy
                scm_mask = steps_mask(
                    total_steps=num_inference_steps,
                    mask_policy=scm_policy,
                )
                config.steps_computation_mask = scm_mask
                config.steps_computation_policy = "dynamic"
            elif skip_interval > 0:
                # Generate custom mask with skip_interval
                scm_mask = _generate_skip_interval_mask(
                    num_inference_steps, skip_interval, max_warmup_steps
                )
                config.steps_computation_mask = scm_mask
                config.steps_computation_policy = "dynamic"
        
        return config
        
    except ImportError as e:
        raise ImportError(f"Failed to build cache config: {e}")


def _generate_skip_interval_mask(
    total_steps: int, 
    skip_interval: int, 
    warmup_steps: int
) -> List[int]:
    """
    Generate steps computation mask with skip_interval.
    
    Forces computation every skip_interval steps for temporal consistency.
    
    Example with total_steps=20, skip_interval=3, warmup=4:
    [1,1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1]
    """
    mask = []
    
    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup: always compute
            mask.append(1)
        elif step == total_steps - 1:
            # Last step: always compute
            mask.append(1)
        elif (step - warmup_steps) % skip_interval == 0:
            # Force compute at interval
            mask.append(1)
        else:
            # Cache
            mask.append(0)
    
    return mask


def build_calibrator_config(taylor_order: int):
    """Build TaylorSeerCalibratorConfig if taylor_order > 0."""
    if taylor_order <= 0:
        return None
    
    try:
        from cache_dit import TaylorSeerCalibratorConfig
        
        return TaylorSeerCalibratorConfig(
            enable_calibrator=True,
            enable_encoder_calibrator=True,
            taylorseer_order=taylor_order,
        )
    except ImportError:
        logger.warning("TaylorSeerCalibratorConfig not available")
        return None


# =============================================================================
# BlockAdapter Construction - Manual Block Extraction for ComfyUI Models
# =============================================================================

def _manual_extract_blocks(transformer: torch.nn.Module) -> Optional[List[torch.nn.Module]]:
    """
    Manually extract transformer blocks from ComfyUI models.
    
    This is necessary because cache-dit's auto-detection fails on non-diffusers
    architectures like NextDiT (Z-Image), Flux (including Flux.2), LTX-2, etc.
    
    Returns:
        List of blocks if found, None otherwise
    """
    blocks = []
    transformer_class = transformer.__class__.__name__.lower()
    
    # Strategy 1: Z-Image (NextDiT architecture)
    # These models store blocks in .layers attribute
    if hasattr(transformer, 'layers'):
        layers = transformer.layers
        if isinstance(layers, (list, torch.nn.ModuleList)):
            blocks = list(layers)
            return blocks
        elif isinstance(layers, torch.nn.Sequential):
            blocks = list(layers.children())
            return blocks
    
    # Strategy 2: Flux (dual-block architecture)
    # Flux has .double_blocks and .single_blocks
    if hasattr(transformer, 'double_blocks') or hasattr(transformer, 'single_blocks'):
        if hasattr(transformer, 'double_blocks'):
            double_blocks = transformer.double_blocks
            if isinstance(double_blocks, (list, torch.nn.ModuleList)):
                blocks.extend(list(double_blocks))
        if hasattr(transformer, 'single_blocks'):
            single_blocks = transformer.single_blocks
            if isinstance(single_blocks, (list, torch.nn.ModuleList)):
                blocks.extend(list(single_blocks))
        if blocks:
            return blocks
    
    # Strategy 3: LTX-2 / HunyuanVideo / Standard DiT
    # These models typically have .blocks or .transformer_blocks
    # LTX-2 note: Uses standard .transformer_blocks attribute
    # - Each block is an LTX2TransformerBlock that handles dual-path processing
    # - Block forward: (h, audio_h, enc_h, audio_enc_h) -> (h, audio_h)
    # - Extracted blocks are used by lightweight cache (not BlockAdapter due to signature mismatch)
    for attr_name in ['blocks', 'transformer_blocks', 'dit_blocks']:
        if hasattr(transformer, attr_name):
            attr_blocks = getattr(transformer, attr_name)
            if isinstance(attr_blocks, (list, torch.nn.ModuleList)):
                blocks = list(attr_blocks)
                logger.info(f"[CacheDiT] ✓ Found {len(blocks)} blocks in .{attr_name}")
                return blocks
            elif isinstance(attr_blocks, torch.nn.Sequential):
                blocks = list(attr_blocks.children())
                logger.info(f"[CacheDiT] ✓ Found {len(blocks)} blocks in .{attr_name} Sequential")
                return blocks
    
    # Strategy 4: Deep search in named_children
    # Last resort: search for ModuleList or Sequential containing blocks
    logger.info(f"[CacheDiT] Standard attributes not found, performing deep search...")
    for name, module in transformer.named_children():
        if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            # Check if this looks like a block container
            children = list(module.children()) if isinstance(module, torch.nn.Sequential) else list(module)
            if len(children) >= 4:  # Reasonable number of blocks
                logger.info(f"[CacheDiT] ✓ Found {len(children)} blocks in .{name} (deep search)")
                return children
    
    logger.warning(f"[CacheDiT] ⚠ Manual block extraction failed - no standard block attributes found")
    return None


def build_block_adapter(
    transformer: torch.nn.Module,
    forward_pattern: str,
    auto_detect: bool = True,
):
    """
    Build BlockAdapter for a transformer model with manual block extraction fallback.
    
    Args:
        transformer: The diffusion model transformer
        forward_pattern: Pattern name (Pattern_0 to Pattern_5)
        auto_detect: Auto-detect transformer blocks (will fallback to manual if fails)
    """
    try:
        from cache_dit import BlockAdapter
        
        pattern = get_forward_pattern(forward_pattern)
        
        # Log transformer info for debugging
        transformer_class = transformer.__class__.__module__ + "." + transformer.__class__.__name__
        logger.info(f"[CacheDiT] Building BlockAdapter for: {transformer_class}")
        
        # CRITICAL FIX: For ComfyUI models, manually extract and inject blocks
        # This ensures cache-dit can find them when enable_cache(transformer) is called
        manual_blocks = _manual_extract_blocks(transformer)
        
        if manual_blocks and len(manual_blocks) > 0:
            logger.info(f"[CacheDiT] ✓ Manual extraction successful: {len(manual_blocks)} blocks")
            
            # Store blocks in transformer for cache-dit to discover
            # Use standard attribute names that cache-dit recognizes
            if not hasattr(transformer, 'blocks'):
                transformer.blocks = torch.nn.ModuleList(manual_blocks)
                logger.info(f"[CacheDiT] Injected blocks into transformer.blocks")
            
            # Create adapter for validation only
            adapter = BlockAdapter(
                transformer=transformer,
                forward_pattern=pattern,
                auto=True,  # Now it can auto-detect from transformer.blocks
            )
        else:
            # Fallback to auto-detection (for diffusers models)
            logger.info(f"[CacheDiT] Manual extraction failed, attempting auto-detection...")
            adapter = BlockAdapter(
                transformer=transformer,
                forward_pattern=pattern,
                auto=auto_detect,
            )
        
        # Verify adapter was created successfully
        if hasattr(adapter, 'blocks') and adapter.blocks:
            logger.info(f"[CacheDiT] ✓ BlockAdapter created successfully with {len(adapter.blocks)} blocks")
        else:
            logger.error(f"[CacheDiT] ✗ BlockAdapter created but blocks list is empty!")
            raise RuntimeError("BlockAdapter has no blocks - caching will not work")
        
        return adapter
        
    except Exception as e:
        logger.error(f"[CacheDiT] Failed to build BlockAdapter: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to build BlockAdapter for {transformer.__class__.__name__}: {e}")


# =============================================================================
# Summary Statistics & ASCII Dashboard
# =============================================================================

def format_summary_dashboard(
    stats: Dict[str, Any],
    model_type: str,
    num_steps: int,
    config_info: Dict[str, Any],
) -> str:
    """
    Format cache-dit summary statistics as an ASCII dashboard.
    
    Returns a beautifully formatted string for terminal/log output.
    """
    if not stats or stats.get("total_steps", 0) <= 0:
        width = 66
        lines = []
        lines.append("╔" + "═" * (width - 2) + "╗")
        lines.append("║" + "CacheDiT Summary: No run statistics".center(width - 2) + "║")
        lines.append("╠" + "═" * (width - 2) + "╣")
        lines.append("║" + " Transformer executed steps were not captured.".ljust(width - 2) + "║")
        lines.append("║" + " Check:".ljust(width - 2) + "║")
        lines.append("║" + "   1. Cache is enabled for this model".ljust(width - 2) + "║")
        lines.append("║" + "   2. Sampling completed its last step".ljust(width - 2) + "║")
        lines.append("║" + "   3. Check logs for cache-dit errors".ljust(width - 2) + "║")
        lines.append("╚" + "═" * (width - 2) + "╝")
        return "\n".join(lines)
    
    # Extract key metrics
    total_steps = stats.get("total_steps", num_steps)
    cached_steps = stats.get("cached_steps", 0)
    computed_steps = stats.get("computed_steps", total_steps - cached_steps)
    cache_hit_rate = (cached_steps / max(total_steps, 1)) * 100
    
    avg_diff = stats.get("avg_residual_diff", 0.0)
    max_diff = stats.get("max_residual_diff", 0.0)
    speedup = stats.get("speedup", total_steps / max(computed_steps, 1))
    
    # Build ASCII table
    width = 66
    lines = []
    
    # Header
    lines.append("╔" + "═" * (width - 2) + "╗")
    title = "CacheDiT Performance Dashboard"
    lines.append("║" + title.center(width - 2) + "║")
    lines.append("╠" + "═" * (width - 2) + "╣")
    
    # Model info section
    lines.append("║" + f"  Model: {model_type}".ljust(width - 2) + "║")
    lines.append("║" + f"  Pattern: {config_info.get('pattern', 'N/A')}".ljust(width - 2) + "║")
    lines.append("║" + f"  Strategy: {config_info.get('strategy', 'N/A')}".ljust(width - 2) + "║")
    lines.append("╠" + "─" * (width - 2) + "╣")
    
    # Performance metrics
    lines.append("║" + "  📊 Performance Metrics".ljust(width - 2) + "║")
    lines.append("║" + "─" * (width - 4) + "  ║")
    
    # Create metric rows
    metrics = [
        ("Total Steps", f"{total_steps}"),
        ("Computed Steps", f"{computed_steps}"),
        ("Cached Steps", f"{cached_steps}"),
        ("Cache Hit Rate", f"{cache_hit_rate:.1f}%"),
        ("Estimated Speedup", f"{speedup:.2f}x"),
    ]
    
    for label, value in metrics:
        row = f"  {label}:".ljust(25) + f"{value}".rjust(width - 29)
        lines.append("║" + row + "║")
    
    lines.append("╠" + "─" * (width - 2) + "╣")
    
    # Quality metrics
    lines.append("║" + "  Quality Metrics".ljust(width - 2) + "║")
    lines.append("║" + "─" * (width - 4) + "  ║")
    
    quality_metrics = [
        ("Threshold", f"{config_info.get('threshold', 0):.4f}"),
        ("Avg Residual Diff", f"{avg_diff:.6f}"),
        ("Max Residual Diff", f"{max_diff:.6f}"),
        ("Fn/Bn Blocks", f"F{config_info.get('fn', 0)}B{config_info.get('bn', 0)}"),
    ]
    
    for label, value in quality_metrics:
        row = f"  {label}:".ljust(25) + f"{value}".rjust(width - 29)
        lines.append("║" + row + "║")
    
    # Advanced settings if present
    if config_info.get("skip_interval", 0) > 0 or config_info.get("noise_scale", 0) > 0:
        lines.append("╠" + "─" * (width - 2) + "╣")
        lines.append("║" + "   Advanced Settings".ljust(width - 2) + "║")
        lines.append("║" + "─" * (width - 4) + "  ║")
        
        if config_info.get("skip_interval", 0) > 0:
            row = f"  Skip Interval:".ljust(25) + f"{config_info['skip_interval']}".rjust(width - 29)
            lines.append("║" + row + "║")
        if config_info.get("noise_scale", 0) > 0:
            row = f"  Noise Scale:".ljust(25) + f"{config_info['noise_scale']:.6f}".rjust(width - 29)
            lines.append("║" + row + "║")
        if config_info.get("taylor_order", 0) > 0:
            row = f"  TaylorSeer Order:".ljust(25) + f"{config_info['taylor_order']}".rjust(width - 29)
            lines.append("║" + row + "║")
    
    # Speedup visualization bar
    lines.append("╠" + "─" * (width - 2) + "╣")
    lines.append("║" + "  Speedup Visualization".ljust(width - 2) + "║")
    
    bar_width = width - 12
    filled = int((speedup - 1.0) / 2.0 * bar_width)  # Scale 1x-3x to bar
    filled = max(0, min(filled, bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    lines.append("║" + f"  [{bar}]  ║")
    lines.append("║" + f"  1.0x".ljust(bar_width // 2) + f"3.0x".rjust(bar_width // 2 + 6) + "  ║")
    
    # Footer with troubleshooting tips
    lines.append("╠" + "═" * (width - 2) + "╣")
    
    # Add troubleshooting tips if cache hit rate is 0
    if cache_hit_rate == 0 and total_steps > 0:
        lines.append("║" + " TROUBLESHOOTING: Cache Hit Rate is 0%".center(width - 2) + "║")
        lines.append("╠" + "─" * (width - 2) + "╣")
        lines.append("║" + "  Possible fixes:".ljust(width - 2) + "║")
        lines.append("║" + f"  • Increase threshold (current: {config_info.get('threshold', 0):.4f})".ljust(width - 2) + "║")
        lines.append("║" + "  • Try threshold 0.15-0.25 for more caching".ljust(width - 2) + "║")
        lines.append("║" + f"  • Reduce Fn blocks (current: F{config_info.get('fn', 0)})".ljust(width - 2) + "║")
        lines.append("║" + "  • Check if blocks were detected (see logs)".ljust(width - 2) + "║")
        lines.append("╠" + "═" * (width - 2) + "╣")
    
    tip = "💡 Lower threshold = better quality, less speedup"
    lines.append("║" + tip.center(width - 2) + "║")
    lines.append("╚" + "═" * (width - 2) + "╝")
    
    return "\n".join(lines)


def get_summary_stats(transformer: torch.nn.Module) -> Dict[str, Any]:
    """
    Get summary statistics from cache-dit.
    
    Returns dict with: total_steps, cached_steps, computed_steps,
    avg_residual_diff, max_residual_diff, speedup
    """
    try:
        import cache_dit
        
        stats = cache_dit.summary(transformer)
        
        if stats is None:
            logger.warning("[CacheDiT] cache_dit.summary() returned None - cache may not be active")
            return {}
        
        # cache_dit.summary() returns List[CacheStats]; the transformer-level
        # entry is the last item and is what upstream uses for overall totals.
        if not isinstance(stats, (list, tuple)):
            stats = [stats]
        stats = [s for s in stats if s is not None]
        if not stats:
            logger.warning("[CacheDiT] cache_dit.summary() returned no stats")
            return {}

        overall = stats[-1]

        def _get(stat, name, default):
            if isinstance(stat, dict):
                return stat.get(name, default)
            return getattr(stat, name, default)

        total_steps = int(_get(overall, "accumulated_transformer_executed_steps", 0) or 0)
        cached_steps = int(_get(overall, "accumulated_cached_steps", 0) or 0)
        cached_steps += int(_get(overall, "cfg_accumulated_cached_steps", 0) or 0)
        computed_steps = max(total_steps - cached_steps, 0)

        residual_diffs = []
        for stat in stats:
            for diff in (_get(stat, "residual_diffs", {}) or {}).values():
                if isinstance(diff, (list, tuple)):
                    residual_diffs.extend(float(v) for v in diff)
                else:
                    residual_diffs.append(float(diff))
            for diff in (_get(stat, "cfg_residual_diffs", {}) or {}).values():
                if isinstance(diff, (list, tuple)):
                    residual_diffs.extend(float(v) for v in diff)
                else:
                    residual_diffs.append(float(diff))

        avg_diff = float(sum(residual_diffs) / len(residual_diffs)) if residual_diffs else 0.0
        max_diff = float(max(residual_diffs)) if residual_diffs else 0.0
        speedup = total_steps / computed_steps if total_steps > 0 and computed_steps > 0 else 1.0

        result = {
            "total_steps": total_steps,
            "cached_steps": cached_steps,
            "computed_steps": computed_steps,
            "avg_residual_diff": avg_diff,
            "max_residual_diff": max_diff,
            "speedup": speedup,
            "raw": stats,
        }
        
        # Debug log if no caching occurred
        if result["total_steps"] > 0 and result["cached_steps"] == 0:
            logger.warning(
                f"[CacheDiT] No steps were cached! "
                f"Check threshold - it may be too strict."
            )
        
        return result
        
    except Exception as e:
        logger.error(f"[CacheDiT] Failed to get summary stats: {e}")
        import traceback
        traceback.print_exc()
        return {}


def print_summary_to_log(
    transformer: torch.nn.Module,
    model_type: str,
    num_steps: int,
    config_info: Dict[str, Any],
) -> str:
    """
    Get summary, format as dashboard, and print to log.
    Returns the formatted string.
    """
    stats = get_summary_stats(transformer)
    dashboard = format_summary_dashboard(stats, model_type, num_steps, config_info)
    
    # ComfyUI logging already mirrors logger output to console; printing again
    # produces a second dashboard in the same run.
    logger.info("\n" + dashboard)
    
    return dashboard


# =============================================================================
# Noise Injection Utility
# =============================================================================

def apply_noise_injection(
    output: torch.Tensor,
    noise_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply small noise perturbation to cached output.
    
    Prevents "static" or "dead" regions in generated content.
    Typical scale: 0.001 - 0.003
    """
    if noise_scale <= 0:
        return output
    
    # Handle generator compatibility (added in PyTorch 1.11.0)
    if generator is not None:
        try:
            noise = torch.randn_like(output, generator=generator) * noise_scale
        except TypeError:
            # Fallback for older PyTorch versions
            noise = torch.randn_like(output) * noise_scale
    else:
        noise = torch.randn_like(output) * noise_scale
    
    return output + noise
