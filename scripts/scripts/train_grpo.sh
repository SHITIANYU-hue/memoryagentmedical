python -m src.training.rlhf.grpo_loop \
  --policy checkpoints/sft \
  --rm     checkpoints/rm \
  --cfg    configs/rlhf_grpo.yaml