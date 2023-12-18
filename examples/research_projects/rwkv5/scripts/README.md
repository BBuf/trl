# DPO pipeline for the creation of Stack RWKV5 : a Stack exchange rwkv-5-1b5 model

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

Since we will use `accelerate` for training, make sure to run:
```
$ accelerate config
```

## Training

There were two main steps to the DPO training process:
1. Supervised fine-tuning of the base rwkv-5-1b5 model to create rwkv-5-1b5-se:
    - `accelerate launch examples/research_projects/rwkv5/scripts/sft_rwkv5.py --training_args.output_dir="sft"`
1. Run the DPO trainer using the model saved by the previous step:
    - `accelerate launch examples/research_projects/rwkv5/scripts/dpo_rwkv5.py --model_name_or_path="sft/final_checkpoint" --output_dir="dpo"`


## Merging the adaptors

To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

```
python examples/research_projects/rwkv5/scripts/merge_peft_adapter.py --base_model_name="RWKV/rwkv-5-world-1b5" --adapter_model_name="dpo/final_checkpoint/" --output_name="stack-rwkv-1b5"
```

which will also push the model to your HuggingFace hub account.

## Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and load them via:

```py
from peft import AutoPeftModelForCausalLM


model = AutoPeftModelForCausalLM.from_pretrained(
    "dpo/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

model.generate(...)
```
