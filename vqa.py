import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
class Config:
    # IN_DIR = Path("/kaggle/input/vqadata/vqa/data/")
    kfold = 10
    fold = 0
    vit_name = "google/vit-base-patch16-224-in21k"
    models = ["google/mt5-base"]
    lm_name = models[0]
    batch_size = 16
    num_train_epochs = 10
    eval_in_epoch = 2
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    learning_rate = 5e-4
    max_grad_norm = 3.0
    warmup_ratio = 0.2
    freeze_embeddings = False
    dropout_rate = 0.1
    output_dir = f'vqa-checkpoints-vitpatch16-t5-rmdup-ver2'

CFG = Config()

import random
import torch
import numpy as np
import json
import pandas as pd
from datasets import Image, Dataset
from sklearn.model_selection import GroupKFold

from transformers import ViltProcessor, ViltFeatureExtractor, AutoModel, AutoTokenizer, ViltModel, ViltConfig, AutoConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import BertConfig, ViTConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import ViltProcessor, ViltForQuestionAnswering

import torch, gc
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
# os.environ["WANDB_DISABLED"] = "true"
import warnings
from augment_dataset import df_to_dataset
warnings.simplefilter(action='ignore', category=FutureWarning)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def read_df(image_folder, data_file):
    f = open(data_file)
    data = json.load(f)

    img_df = pd.DataFrame.from_records(data['images'])
    ann_df = pd.DataFrame.from_records(data['annotations'])
    img_df.rename(columns={"id": "image_id"}, inplace= True)
    img_df['image'] = img_df['filename'].apply(
        lambda x: f"{image_folder}/{x}"
    )
    df = pd.merge(img_df, ann_df, on="image_id")
    return df

# def df_to_dataset(df):
#     dataset = Dataset.from_pandas(df, preserve_index = False).cast_column("image", Image())
#     dataset = dataset.remove_columns(['filename'])
#     return dataset

df = read_df('data/train-images', 'data/evjvqa_train.json')
wrong_data = [1493, 2397, 2900, 2913, 2952, 2955, 2956, 2959, 2989, 4094]
df = df[~df.id.isin(wrong_data)].reset_index(drop=True)

duplicate_data = [10008,10009,10012,10082,10083,10013,10014,10015,10088,10018,10089,10091,10022,10092,10093,10023,10024,10097,10028,10098,10099,10101,10030,10031,10033,10104,10034,10035,10108,10109,10110,10111,10112,10113,10114,10115,10044,10048,10117,10118,10119,10050,10051,10121,10052,10125,10054,10056,10127,10129,10130,10131,10132,10133,10134,10064,10065,10135,10138,10068,10069,10139,10071,10142,10144,10145,10146,10075,10148,10149,10150,10151,18584,20381,21394]
df = df[~df.id.isin(duplicate_data)].reset_index(drop=True)

gkf = GroupKFold(n_splits=CFG.kfold)
for fold, ( _, val_) in enumerate(gkf.split(X=df, groups=df.image_id)):
    df.loc[val_ , "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
train_df = df[df["kfold"] != CFG.fold]
eval_df = df[df["kfold"] == CFG.fold]

from polyglot.detect import Detector
train_df['lang'] = train_df.copy().question.apply(lambda text: Detector(text).language.code)
eval_df['lang'] = eval_df.copy().question.apply(lambda text: Detector(text).language.code)

print(train_df.head())

train_ds = df_to_dataset(train_df, augment= True).remove_columns(['kfold'])
eval_ds = df_to_dataset(eval_df).remove_columns(['kfold'])
print('train_ds', len(train_ds))
print('eval_ds', len(eval_ds))

test_df = read_df('data/public-test-images', 'data/evjvqa_public_test.json')
test_df['lang'] = test_df.question.apply(lambda text: Detector(text).language.code)
test_ds = df_to_dataset(test_df)\
print('test_ds', len(test_ds))

# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0
# lang_map = {
#     'en': 'english',
#     'vi': 'vietnamese',
#     'ja': 'japanese',
#     'ko': 'japanese',
#     'af': 'english',
#     'tl': 'english',
#     'no': 'english',
#     'so': 'english',
# }
# def which_language(row):
#     lang = detect(row['question'])
#     out = {
#         'lang': None
#     }
#     try:
#         out['lang'] = lang_map[lang]
#     except:
#         raise ValueError("Unknown language: " + lang + row['question'])
#     return out

# train_ds = train_ds.map(which_language)
# test_ds = test_ds.map(which_language)
# eval_ds = eval_ds.map(which_language)

from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained(CFG.lm_name)
tokenizer = T5Tokenizer.from_pretrained(CFG.lm_name)
model.encoder.main_input_name = 'inputs_embeds'

from transformers import ViTFeatureExtractor, ViTModel
import torch
from datasets import load_dataset

feature_extractor = ViTFeatureExtractor.from_pretrained(CFG.vit_name)
vit = ViTModel.from_pretrained(CFG.vit_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import copy
embed_model = copy.deepcopy(model.shared)
embed_model = embed_model.to(device)
vit = vit.to(device)


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f} Mb'.format(size_all_mb))
num_params = sum(p.numel() for p in model.parameters())
print(f'Num params: {num_params/(1e6):.3f} M')


question_length = 70
answer_length = 70

import torch
from torch.utils.data import DataLoader

class DataCollator:
    def __init__(self, img_model, feature_extractor, tokenizer, text_embed_model):
        self.img_model = img_model
        self.tokenizer = tokenizer
        self.text_embed_model = text_embed_model
        self.feature_extractor = feature_extractor
        
    def __call__(self, batch):
        # tokenize the inputs and labels
        image = [i['image'] for i in batch]
        question = [i['lang'] + ': ' + i['question'] for i in batch]
#         question = [i['question'] for i in batch]
        answer = [i['answer'] for i in batch]
        ques_inputs = self.tokenizer(question, max_length = question_length, padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            ques_embeds = self.text_embed_model(ques_inputs['input_ids'].to(device)).cpu().detach()
        image_inputs = self.feature_extractor(image, return_tensors="pt")
        for u, v in image_inputs.items():
            image_inputs[u] = v.to(device)
        img_embeds = self.img_model(**image_inputs).last_hidden_state.cpu().detach()
        inputs_embeds = torch.cat((ques_embeds,img_embeds),1)
        
        
        clear_gpu()
        attention_mask = torch.cat((ques_inputs.attention_mask,torch.ones(img_embeds.shape[0],img_embeds.shape[1])),1)
        del image_inputs, ques_embeds, img_embeds
        outputs = self.tokenizer(answer, max_length = answer_length, padding='max_length', truncation=True, return_tensors='pt')
        labels = outputs.input_ids.clone()
#         labels = torch.where(labels== tokenizer.pad_token_id, -100, labels)
        labels[labels == tokenizer.pad_token_id] = -100

        result = {}
        result["labels"] = labels
        result["inputs_embeds"] = inputs_embeds
        result["attention_mask"] = attention_mask
#         result["decoder_input_ids"] = outputs.input_ids 
#         result["decoder_attention_mask"] = outputs.attention_mask 
        return result

class TestCollator:
    def __init__(self, img_model, feature_extractor, tokenizer, text_embed_model):
        self.img_model = img_model
        self.tokenizer = tokenizer
        self.text_embed_model = text_embed_model
        self.feature_extractor = feature_extractor
    
    def __call__(self, batch):
        # tokenize the inputs and labels
        image = [i['image'] for i in batch]
        question = [i['lang'] + ': ' + i['question'] for i in batch]
#         question = [i['question'] for i in batch]
        answer = [i['answer'] for i in batch]
        ques_inputs = self.tokenizer(question, max_length = question_length, padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            ques_embeds = self.text_embed_model(ques_inputs['input_ids'].to(device)).cpu().detach()
            image_inputs = self.feature_extractor(image, return_tensors="pt")
            for u, v in image_inputs.items():
                image_inputs[u] = v.to(device)
            img_embeds = self.img_model(**image_inputs).last_hidden_state.cpu().detach()
            inputs_embeds = torch.cat((ques_embeds,img_embeds),1) 
        
        clear_gpu()
        attention_mask = torch.cat((ques_inputs.attention_mask,torch.ones(img_embeds.shape[0],img_embeds.shape[1])),1)
        del image_inputs, ques_embeds, img_embeds
        
        result = {}
        result["inputs_embeds"] = inputs_embeds
        result["attention_mask"] = attention_mask

        return result
        
train_collator = DataCollator(vit, feature_extractor, tokenizer, embed_model)
test_collator = TestCollator(vit, feature_extractor, tokenizer, embed_model)

import evaluate
from evaluate_metrics import compute_f1, compute_avg_bleu


def compute_metrics(eval_preds):
    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_json = {}
    label_json = {}
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        pred_json[i] = pred
        label_json[i] = label
    result = {
        'f1': compute_f1(pred_json, label_json),
        'bleu': compute_avg_bleu(pred_json, label_json)
    }

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    label_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
    result["gen_len"] = np.mean(prediction_lens)
    result["label_len"] = np.mean(label_lens)
    
    result = {k: round(v, 4) for k, v in result.items()}
    length = len(decoded_preds)
    for idx in np.random.randint(0,length, size=10):
        print('-'*35)
        print(f"{idx} - Label: {decoded_labels[idx]}")
        print(f"{idx} - Predict: {decoded_preds[idx]}")
        
    return result

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

batch_size = CFG.batch_size
num_train_epochs = CFG.num_train_epochs
eval_in_epoch = CFG.eval_in_epoch
gradient_accumulation_steps = CFG.gradient_accumulation_steps
training_steps = int(len(train_ds)/ (batch_size * gradient_accumulation_steps))
eval_steps = int(training_steps / eval_in_epoch)

print(f'eval_steps: {eval_steps} | training_steps {training_steps}')
clear_gpu()
training_args = Seq2SeqTrainingArguments(
    output_dir=CFG.output_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    num_train_epochs=num_train_epochs,
#     optim = 'adamw_torch',
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=eval_steps,
    save_steps=eval_steps,
    eval_steps=eval_steps,
    # warmup_steps=0,
    overwrite_output_dir=True,
    fp16=False,
    report_to = 'none',
    remove_unused_columns = False,
    save_total_limit=3,
    # load_best_model_at_end = True
)
training_args.weight_decay = CFG.weight_decay
training_args.learning_rate = CFG.learning_rate
training_args.max_grad_norm = CFG.max_grad_norm
training_args.warmup_ratio = CFG.warmup_ratio

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator = train_collator
)
# clear_gpu()
trainer.train()
trainer.evaluate(test_ds)
# clear_gpu()
trainer.save_state()
from tqdm import tqdm
def get_predict(test_ds):
    clear_gpu()
    test_loader = DataLoader(test_ds, batch_size = 32, collate_fn = test_collator, shuffle = False)
    outputs = []
    for inputs in tqdm(test_loader):
        for u, v in inputs.items():
            inputs[u] = v.to(device)
        result = model.generate(**inputs)
        decoded = tokenizer.batch_decode(result, skip_special_tokens=True)
        outputs += decoded
    # for idx in range(len(outputs)):
    #     print('-'*35)
    #     print(f"{idx} - Label: {test_ds['answer'][idx]}")
    #     print(f"{idx} - Predict: {outputs[idx]}")
    clear_gpu()
    return outputs

outputs = get_predict(test_ds)

id_list = test_ds['id']
result = {}
for idx, out in zip(id_list,outputs):
    result[idx] = out

import json
 
# Serializing json
json_object = json.dumps(result, indent=4,ensure_ascii=False)
 
# Writing to sample.json
with open("results.json", "w",encoding='utf8') as outfile:
    outfile.write(json_object)
