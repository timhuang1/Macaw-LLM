import os
import pickle
import json
import codecs
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import numpy as np
import torch

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    ),
}

coco_examples_filename = "generated_examples_coco.json"
avsd_examples_filename = "generated_examples_avsd.json"
alpaca_examples_filename = "alpaca_data.json"
visual_names_filename = "all_visual_names_instruction.json"
saved_ds_filename = "train_total_new_instruction.cache.new"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Building and processing datasets for mllm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="The raw data dir. Should contain sub-folders, including /avsd, /vqa, /alpaca_data."
    )
    parser.add_argument(
        "--default_sample_number", type=int, default=50000, help="Default number of instances to sample for each subset"
    )
    parser.add_argument(
        "--vqa_sample_number", type=int, default=None, help="Number of instances to sample for vqa dataset"
    )
    parser.add_argument(
        "--alpaca_sample_number", type=int, default=None, help="Number of instances to sample for alpaca dataset"
    )
    parser.add_argument(
        "--avsd_sample_number", type=int, default=None, help="Number of instances to sample for avsd dataset"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )

    args = parser.parse_args()
    args.vqa_sample_number = args.vqa_sample_number or args.default_sample_number
    args.alpaca_sample_number = args.alpaca_sample_number or args.default_sample_number
    args.avsd_sample_number = args.avsd_sample_number or args.default_sample_number
    return args


def preprocess_coco_to_tensor_dataset(all_visual_names, tokenizer):
    # all_examples = json_load('data/generated_examples_coco.json')['data']
    all_examples = json_load(os.path.join(args.data_dir, coco_examples_filename))['data']

    max_length = 256
    all_images, all_null_audios, all_null_videos = [], [], []
    all_texts, all_labels = [], []

    all_textual_inputs = []
    all_native_labels = []
    for ind, e in enumerate(tqdm(all_examples)):
        if 'caption' in e['instruction'] or 'caption' in e['response'] or ' no ' in e['response'] or 'not' in e['response']:
            continue
        all_images.append(all_visual_names[e['id']])
        e = {
            'instruction': e['instruction'],
            'input': "",
            'output': e['response']
        }
        texts = PROMPT_DICT['prompt_input'].format(e['instruction'], e['input']) if e['input'] != "" else PROMPT_DICT['prompt_no_input'].format(e['instruction'])
        full_texts = texts + '\n {} \n\n'.format(e['output']) + tokenizer.eos_token_id

        all_textual_inputs.append(full_texts)
        t_all = tokenizer.encode(full_texts)
        
        t_texts = tokenizer.encode(texts)

        if len(t_texts) >= max_length:
            continue
        if len(t_all) > max_length:
            t_all = t_all[:max_length]
        if len(t_all) < max_length:
            t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))

        prefix_len = len(t_texts) - 1
        labels = [IGNORE_INDEX] * prefix_len + t_all[prefix_len:]
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
        all_texts.append(torch.tensor([t_all], dtype=torch.int))
        all_labels.append(torch.tensor([labels], dtype=torch.int))
        all_native_labels.append(labels)

    all_null_audios = [-1] * len(all_images)
    all_null_videos = all_null_audios

    return all_textual_inputs, all_native_labels, all_images, all_null_audios, all_null_videos


def preprocess_alpaca_to_tensor_dataset(tokenizer):
    # all_examples = json_load('data/alpaca_data/alpaca_data.json')
    all_examples = json_load(os.path.join(args.data_dir, alpaca_examples_filename))

    max_length = args.max_length
    all_null_images, all_null_audios, all_null_videos = [], [], []
    all_texts, all_labels = [], []

    all_textual_inputs = []
    all_native_labels = []
    for ind, e in enumerate(tqdm(all_examples)):
        texts = PROMPT_DICT['prompt_input'].format(e['instruction'], e['input']) if e['input'] != "" else PROMPT_DICT['prompt_no_input'].format(e['instruction'])
        full_texts = texts + '\n {} \n\n'.format(e['output']) + tokenizer.eos_token_id
        t_all = tokenizer.encode(full_texts)

        t_texts = tokenizer.encode(texts)
        if len(t_texts) >= max_length:
            continue
        if len(t_all) > max_length:
            t_all = t_all[:max_length]
        if len(t_all) < max_length:
            t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))
        all_textual_inputs.append(full_texts)

        prefix_len = len(t_texts) - 1
        labels = [IGNORE_INDEX] * prefix_len + t_all[prefix_len:]        
        if len(labels) > max_length:
            labels = labels[:max_length]
        if len(labels) < max_length:
            labels = labels + [IGNORE_INDEX] * (max_length - len(labels))    
        
        all_texts.append(t_all)
        all_labels.append(labels)
        all_native_labels.append(labels)

    all_null_images = [-1] * len(all_texts)
    all_null_audios = all_null_images
    all_null_videos = all_null_images 

    return all_textual_inputs, all_native_labels, all_null_images, all_null_audios, all_null_videos


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def preprocess_avsd_to_tensor_dataset(all_visual_names, tokenizer):
    # train_metadata_dir = 'data/generated_examples_avsd.json'
    train_metadata_dir = os.path.join(args.data_dir, avsd_examples_filename)

    torch.random.manual_seed(0)
    max_length = args.max_length
    
    def read_image_and_audio(metadata_dir):
        metadata = json_load(metadata_dir)['data']

        all_videos, all_audios, all_texts, all_null_images = [], [], [], []
        all_labels = []

        all_textual_inputs = []
        all_native_labels = []
        for ind, e in enumerate(tqdm(metadata)):
            if 'caption' in e['instruction'] or 'caption' in e['response'] or ' no ' in e['response'] or 'not' in e['response']:
                continue

            prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n {} \n\n"
            q = prompt.format(e['instruction'], e['response']) + + tokenizer.eos_token_id
            t_all = tokenizer.encode(q, max_length=max_length, truncation=True)

            q_input = q.split(' Response:')[0] + ' Response:'

            if len(t_all) > max_length:
                t_all = t_all[:max_length]
            if len(t_all) < max_length:
                t_all = t_all + [tokenizer.pad_token_id] * (max_length - len(t_all))

            len_t_q = len(tokenizer.encode(q_input)) - 1
            labels = [IGNORE_INDEX] * len_t_q + t_all[len_t_q:]
            if len(labels) > max_length:
                labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
            all_textual_inputs.append(q)
            all_native_labels.append(labels)

            all_videos.append(all_visual_names[e['id']])
            all_audios.append(all_visual_names[e['id']])
            all_null_images.append(-1)
            all_texts.append(torch.tensor([t_all], dtype=torch.int))
            all_labels.append(torch.tensor([labels], dtype=torch.int))
        
        return all_textual_inputs, all_native_labels, all_null_images, all_audios, all_videos

    all_textual_inputs, all_native_labels, all_images, all_audios, all_videos = read_image_and_audio(train_metadata_dir)

    return all_textual_inputs, all_native_labels, all_images, all_audios, all_videos


def preprocess_all_datasets(args):
    all_visual_names = json_load(os.path.join(args.data_dir, visual_names_filename))['dict']
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # # Chenyang: 2023-05-21, add special tokens
    # special_tokens_dict = {'additional_special_tokens': ['<image>', '</image>', '<audio>', '</audio>', '<video>', '</video>']}

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right"

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    # tokenizer.save_pretrained('trained_models/llama_tokenizer')

    all_image_data = preprocess_coco_to_tensor_dataset(all_visual_names, tokenizer)
    all_text_data = preprocess_alpaca_to_tensor_dataset(tokenizer)
    all_video_data = preprocess_avsd_to_tensor_dataset(all_visual_names, tokenizer)

    def draw_examples(lis, num):
        ri = draw_samples([i for i in range(len(lis))], num)
        return ri

    ra, rb, rc = None, None, None
    all_dataset = []
    i = 0
    for a, b, c in zip(all_image_data, all_text_data, all_video_data):
        if ra is None:
            print(f"VQA total sample: {len(a)}; Alpaca total sample:{len(b)}; AVSD total sample: {len(c)}")
            ra = draw_examples(a, args.vqa_sample_number)
            rb = draw_examples(b, args.alpaca_sample_number)
            rc = draw_examples(c, args.avsd_sample_number)

            a = [a[i] for i in ra]
            b = [b[i] for i in rb]

            c = [c[i] for i in rc]

            new_lis = a + b + c
            print(len(new_lis))
            all_dataset.append(new_lis)
        else:
            # print(len(a), len(b), len(c))
            a = [a[i] for i in ra]
            b = [b[i] for i in rb]

            c = [c[i] for i in rc]
            new_lis = a + b + c
            # print(len(new_lis))
            all_dataset.append(new_lis)
        i += 1

    max_length = args.max_length
    tokenized_texts = tokenizer(
        all_dataset[0],
        max_length=max_length,
        padding='max_length',
        add_special_tokens=False,
        truncation=True
    )
    tokenized_texts['labels'] = all_dataset[1]
    
    tokenized_texts['images'] = all_dataset[2]
    tokenized_texts['audios'] = all_dataset[3]
    tokenized_texts['videos'] = all_dataset[4]

    for k in tokenized_texts:
        print(k)

    pickle.dump(tokenized_texts, open(os.path.join(args.data_dir, saved_ds_filename), "wb"), protocol=4)


def combine_visual_and_audio_names(args):
    all_names = []

    image_examples = json_load(os.path.join(args.data_dir, coco_examples_filename))['data']
    video_examples = json_load(os.path.join(args.data_dir, avsd_examples_filename))['data']

    for e in image_examples:
        all_names.append(e['id'])
    
    for e in video_examples:
        all_names.append(e['id'])
    
    all_names_dict = {k: ind for ind, k in enumerate(all_names)}
    all_names = {'dict': all_names_dict, 'list': all_names}

    json_dump(all_names, os.path.join(args.data_dir, visual_names_filename))


if __name__ == '__main__':
    args = parse_args()
    combine_visual_and_audio_names(args)
    preprocess_all_datasets(args)
