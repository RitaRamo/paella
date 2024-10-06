from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
from torch import nn

CAPTION_LENGTH = 33
SIMPLE_PREFIX = "A caption I can generate to describe this image in english is: "
LGS = ['ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr', 'iw', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh']

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=1, is_test=False, max_length=None, template_len=0):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = template

    prefix_ids = tokenizer.encode(prefix)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        prefix_ids = prefix_ids[:CAPTION_LENGTH*k + template_len]
        text_ids = text_ids[:CAPTION_LENGTH]
    

    input_ids = prefix_ids + text_ids if not is_test else prefix_ids
    len_prefix = len(prefix_ids)
    
    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    pred = pred.split(":")[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

class TrainDataset(Dataset):
    
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25, multilingual=False, ablation_visual=False):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length + len(tokenizer.encode(SIMPLE_PREFIX))
        self.multilingual=multilingual
        self.templates={}

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.len_of_template= len(tokenizer.encode(self.template))
            self.max_target_length = (CAPTION_LENGTH  # target caption
                                    + CAPTION_LENGTH * k # retrieved captions
                                    + self.len_of_template # template
                                    + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                    )

            if multilingual:
                #using translated/multilingual retrieved captions (instead of english captions)
                for lg in LGS:
                    self.templates[lg]=open("src/templates/template_"+lg+".txt").read().strip() + ' '
            
            
            assert k is not None 
            self.k = k
        else:
            self.template = open(template_path).read().split("||")[1].strip() + ' '
            if multilingual:
                #using translated captions (instead of english captions)
                for lg in LGS:
                    self.templates[lg]=open("src/templates/template_"+lg+".txt").read().split("||")[1].strip() + ' '

                self.max_target_length =CAPTION_LENGTH + len(self.tokenizer.encode(self.template))*2

        self.rag = rag
        self.ablation_visual=ablation_visual
        self.tag="file_name"
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]

        if self.rag: 
            caps = self.df['caps'][idx]
            if self.multilingual:
                lg = self.df['lg'][idx]
                if lg:
                    self.template =  self.templates[lg]

            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                    retrieved_caps=caps, k=self.k, max_length=self.max_target_length, template_len=self.len_of_template)
        else:
            if self.multilingual:
                lg = self.df['lg'][idx]
                self.template =  self.templates.get(lg, self.template)
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template, max_length=self.max_target_length )
       
        encoder_outputs = self.features[str(self.df[self.tag][idx])][()]
        if self.ablation_visual:
            encoder_outputs = torch.zeros((50, 768))

        encoding = {"encoder_outputs": encoder_outputs, 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding



def load_data_for_training_multilingual(annot_path, sampling=False):
    annotations = json.load(open(annot_path))['images']
    
    if sampling is False: 
        #PAELLA_core (includes only en, es, zh and hi)
        core_retrieved_caps={
                "en":json.load(open("data/retrieved_caps/retrieved_caps_resnet50x64.json")),
                "es":json.load(open("data/retrieved_caps/retrieved_ids_resnet50x64.es.json")),
                "zh":json.load(open("data/retrieved_caps/retrieved_ids_resnet50x64.zh.json")),
                "hi":json.load(open("data/retrieved_caps/retrieved_ids_resnet50x64.hi.json"))
            }

    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        samples = []
        for sentence in item['sentences']:
            lg=item.get('lg', None)
            if sampling:
                #PAELLA (the file "dataset_coco_sample_all_35" already includes the sampled retrieved captions for each item)
                caps = item['rags']
            else:
                #PAELLA_core    
                caps = core_retrieved_caps[lg][str(item['cocoid'])]
               
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': sentence['raw'], 'lg': lg})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
    return data 


def load_data_for_training(annot_path, caps_path=None):
    #PALLEA_mono (the english version from SmallCap)
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}
    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 


def load_data_for_inference(annot_path, split, caps_path=None):
    #coco
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:

        #if item['split'] == 'test' or 
        if item['split'] == split:
            file_name = item['filename'].split('_')[-1]
            if caps_path is not None:
                caps = retrieved_caps[str(item['cocoid'])]
            else:
                caps = None
            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
            if item['split'] == 'test':
                data['test'].append(image)
            elif item['split'] == 'val':
                data['val'].append(image)

    return data      


def load_data_for_inference_xm3600(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'val': []}
    for item in annotations:
        file_name = item['file_name'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['id'])]
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['id'])}
        data['val'].append(image)

    return data      
