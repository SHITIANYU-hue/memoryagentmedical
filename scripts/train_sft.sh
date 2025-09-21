python -m src.training.sft.trainer_sft \
  --model Qwen2.5-Med-7B \
  --data data/processed/sft.jsonl \
  --peft qlora --bf16 --batch 4 --grad-accum 8 \
  --save-dir checkpoints/sft