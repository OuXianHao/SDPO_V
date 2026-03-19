# Startup Stall Static Audit (verl/Ray/vLLM/Qwen3-VL)

This note captures static call-chain findings for startup stalling before visible training steps.

## Key findings

- `filter_overlong_prompts=True` triggers a full HuggingFace `Dataset.filter(...)` pass in `RLHFDataset.__init__`.
- For video samples, `_filter_overlong_prompts` calls `process_video(...)` and then `self.processor(...)` to build multimodal model inputs and count `input_ids` length.
- `process_video(...)` calls `qwen_vl_utils.vision_process.fetch_video(...)` with fixed `nframes=32` (with fps fallback), which is expensive and can emit video sampling debug logs.
- Dataloader construction happens before `trainer.init_workers()` and `trainer.fit()`, so full filtering can delay first training step.
- The same video decode path is also used in `__getitem__` and rollout/actor processing stages, so repeated logs can appear beyond filtering too.
