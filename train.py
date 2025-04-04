# %%
# !pip install accelerate==0.33.0
# !pip install bitsandbytes==0.43.3
# !pip install peft==0.12.0 
# !pip install transformers==4.44.0

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

VER=157

# FINAL SOLUTION IS USE_QLORA=FALSE, TRAIN_100_PERCENT=TRUE, ADD_33K=TRUE, DEBUG=FALSE
USE_QLORA = False
TRAIN_100_PERCENT = True
ADD_33K = False
DEBUG = False

# %%
import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score

# %%
@dataclass
class Config:
    output_dir: str = f"output-{VER}"
    checkpoint: str = "gemma2-9b-it-fp16"  
    max_length: int = 2048
    n_splits: int = 5
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # global batch size is 8 
    per_device_eval_batch_size: int = 4
    n_epochs: int = 1
    freeze_layers: int = 0 # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 64
    lora_alpha: float = 4 
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    
config = Config()

# %%
training_args = TrainingArguments(
    output_dir = f"output-{VER}",
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no", # don't save any checkpoints
    #save_steps=200,
    optim=config.optim_type,
    fp16=True, 
    #bf16=False,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,

    #gradient_checkpointing=True, # this doesn't work correctly for some reason

    #logging_first_step=True,
    #lr_scheduler_type='linear', # "cosine" or "linear" or "constant" (default is linear)
    metric_for_best_model='log_loss',
    greater_is_better=False,  
    #save_total_limit=4,
    #load_best_model_at_end=True,
)

# %%
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj",
                    "down_proj","up_proj","o_proj","gate_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score","classifier_head1", "classifier_head2"]
)

# %%
tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

# %%
qlora = {}
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", #nf4 or fp4
        bnb_4bit_use_double_quant = False,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules = ["score","classifier_head1", "classifier_head2"]
    )
    qlora['quantization_config'] = bnb_config
    print("Using QLoRA")

# %%
from torch import nn
import torch.nn.functional as F
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

# %%
import torch
import torch.nn as nn
from transformers import Gemma2ForSequenceClassification, Gemma2Config

class CustomGemma2ForSequenceClassification(Gemma2ForSequenceClassification):
    def __init__(self, config, num_labels_head1=60, num_labels_head2=60):
        super().__init__(config)
        self.num_labels_head1 = num_labels_head1
        self.num_labels_head2 = num_labels_head2
        self.classifier_head1 = nn.Linear(config.hidden_size, num_labels_head1, bias=False)
        self.classifier_head2 = nn.Linear(config.hidden_size, num_labels_head2, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, teacher_logits=None, **kwargs):
        device = input_ids.device

        if labels is not None:
            labels = labels.to(device)
            outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels[:, 0], output_hidden_states=True)
        else:
            outputs = super().forward(input_ids, attention_mask=attention_mask)

        last_token_indices = (torch.sum(attention_mask, dim=1) - 1).to(device)
        last_token_outputs = outputs.hidden_states[-1].to(device)[
            torch.arange(outputs.hidden_states[-1].shape[0], device=device), last_token_indices]

        outputs_head1 = self.classifier_head1(last_token_outputs).to(device)
        outputs_head2 = self.classifier_head2(last_token_outputs).to(device)

        if labels is not None:
            labels_head1 = labels[:, 1].to(device)
            labels_head2 = labels[:, 2].to(device)
            
            dis_loss1 = DistillKL(T=2)(outputs.logits.to(device), teacher_logits[:, :2].to(device))
            dis_loss2 = DistillKL(T=2)(outputs.logits.to(device), teacher_logits[:, 2:].to(device))
            loss_head1 = nn.CrossEntropyLoss()(outputs_head1, labels_head1)
            loss_head2 = nn.CrossEntropyLoss()(outputs_head2, labels_head2)
            loss = outputs.loss.to(device) + 0.1 * loss_head1 + 0.1 * loss_head2 + 0.25 * dis_loss1 + 0.25 * dis_loss2
            return {"loss": loss, "logits": (outputs.logits, outputs_head1, outputs_head2)}
        else:
            return {"logits": (outputs.logits, outputs_head1, outputs_head2)}

config2 = Gemma2Config.from_pretrained(config.checkpoint)
config2.num_labels = 2
model = CustomGemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    config=config2,
    num_labels_head1=126,
    num_labels_head2=126,
    torch_dtype=torch.float16,
    device_map="auto",
    **qlora
)

model.config.use_cache = False
model.config.attn_logit_softcapping = None
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model

# %%
model.print_trainable_parameters()

# %%
import pandas as pd

df = pd.read_parquet("dataset/10k_train.parquet")
df_m0 = pd.DataFrame(np.load('dataset/logits_m0.npy')[:, :2], columns=['m0_a', 'm0_b'])
df_m3 = pd.DataFrame(np.load('dataset/logits_m3.npy')[:, :2], columns=['m3_a', 'm3_b'])
df = pd.concat([df, df_m0, df_m3], axis=1)
df["id"] = df["id"].astype("str")
print('Competition data has shape', df.shape )
LN = len(df)
df.head(1)

# %%
if ADD_33K:
    df = pd.concat([df,df2],axis=0,ignore_index=True)
if DEBUG:
    df = df.iloc[:64].copy()
print("We will use train data with shape", df.shape )

# %%
import numpy as np
m1 = df.model_a.unique()
m2 = df.model_b.unique()
m = np.union1d(m1,m2)
m = sorted(m)
print(f"There are {len(m)} unique models:")

MAP = {x:y for x,y in zip(m,range(len(m)))}
print(MAP)

df.model_a = df.model_a.map(MAP).astype('int32')
df.model_b = df.model_b.map(MAP).astype('int32')
df.head(1)

# %%
ds = Dataset.from_pandas(df)

# %%
import json

class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_text(self, prompts, responses_a, responses_b):

        prompt_length = len(self.tokenizer(prompts)['input_ids'])
        if prompt_length > self.max_length // 2:
            prompt_length = self.max_length // 2
            prompts = self.tokenizer.decode(self.tokenizer(prompts)['input_ids'][-prompt_length:-1])
        response_a_length = len(self.tokenizer(responses_a)['input_ids'])
        response_b_length = len(self.tokenizer(responses_b)['input_ids'])
        if prompt_length + response_a_length + response_b_length > self.max_length:
            response_length = (self.max_length - prompt_length) // 2
            responses_a = self.tokenizer.decode(self.tokenizer(responses_a)['input_ids'][-response_length:-1])
            responses_b = self.tokenizer.decode(self.tokenizer(responses_b)['input_ids'][-response_length:-1])
        
        rounds = [
            f"<start_of_turn>prompt\n{prompts}<end_of_turn>\n"
            +f"<start_of_turn>response_a\n{responses_a}<end_of_turn>\n"
            +f"<start_of_turn>response_b\n{responses_b}<end_of_turn>"
        ]
        
        # for k in range(len(rounds)):
        #     tmp = "\n".join(rounds[k:])
        #     if len( self.tokenizer(tmp)["input_ids"] ) < self.max_length: 
        #         break
        tmp = rounds[0]
        
        return tmp
        
    def __call__(self, batch: dict) -> dict:
        
        texts = [
            self.prepare_text(p, r_a, r_b)
            for p, r_a, r_b in zip(batch["prompt"], batch["response_a"], batch["response_b"])
        ]
        
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels=[]
        for win, c, d in zip(batch["winner"], 
                                   batch["model_a"],batch["model_b"]):
            if win == 'model_a':
                label = 0
            elif win == 'model_b':
                label = 1
            labels.append( (label,c,d) )
        teacher_logits = []
        for m0_a, m0_b, m3_a, m3_b in zip(batch['m0_a'], batch['m0_b'], batch['m3_a'], batch['m3_b']):
            teacher_logits.append((m0_a, m0_b, m3_a, m3_b))
        return {**tokenized, "labels": labels, "teacher_logits": teacher_logits}#, "texts": texts}

# %%
encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True, num_proc=8)

# %%
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = np.array( eval_preds.label_ids )
    
    # Split the predictions and labels into two heads
    preds_head1 = preds[0]
    preds_head2 = preds[1]
    preds_head3 = preds[2]
    labels_head1 = labels[:,0]
    labels_head2 = labels[:,1]
    labels_head3 = labels[:,2]
    
    # Compute log loss and accuracy for each head
    probs_head1 = torch.from_numpy(preds_head1).float().softmax(-1).numpy()
    loss_head1 = log_loss(y_true=labels_head1, y_pred=probs_head1, labels=[x for x in range(2)])
    acc_head1 = accuracy_score(y_true=labels_head1, y_pred=preds_head1.argmax(-1))
    
    probs_head2 = torch.from_numpy(preds_head2).float().softmax(-1).numpy()
    loss_head2 = log_loss(y_true=labels_head2, y_pred=probs_head2, labels=[x for x in range(126)])
    acc_head2 = accuracy_score(y_true=labels_head2, y_pred=preds_head2.argmax(-1))

    probs_head3 = torch.from_numpy(preds_head3).float().softmax(-1).numpy()
    loss_head3 = log_loss(y_true=labels_head3, y_pred=probs_head3, labels=[x for x in range(126)])
    acc_head3 = accuracy_score(y_true=labels_head3, y_pred=preds_head3.argmax(-1))
    
    # Return the metrics for each head
    return {
        "acc_classify": acc_head1,
        "log_loss_classify": loss_head1,
        "acc_model_a": acc_head2,
        "log_loss_model_a": loss_head2,
        "acc_model_b": acc_head3,
        "log_loss_model_b": loss_head3
    }

# %%
if TRAIN_100_PERCENT:
    folds = [
        (
            [i for i in range(len(ds))], 
            [i for i in range(len(ds)) if (i % config.n_splits == fold_idx)&(i<LN)]
        ) 
        for fold_idx in range(config.n_splits)
    ]
    print("We are training with 100% data")
else:
    folds = [
        (
            [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
            [i for i in range(len(ds)) if (i % config.n_splits == fold_idx)&(i<LN)]
        ) 
        for fold_idx in range(config.n_splits)
    ]    

# %%
train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args, 
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()

# %%
trainer.save_model(f"LoRA-v{VER}")


