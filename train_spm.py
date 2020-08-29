import sentencepiece as spm
import os, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--raw_exists', default=False, action='store_true')
args = parser.parse_args()

if not args.raw_exists:
    with open(os.path.join(args.input, "train.json"), 'r', encoding='utf-8') as f:
            chat_dict = json.load(f)

    with open(os.path.join(args.input, "source.txt"), "w") as src_file:
        with open(os.path.join(args.input, "tgt.txt"), "w") as tgt_file:
            for chat in chat_dict.values():
                for turn in chat:
                    src = turn['source']
                    src_file.write(f"{src}\n")
                    tgt = turn['target']
                    tgt_file.write(f"{tgt}\n")

spm.SentencePieceTrainer.train(input=os.path.join(args.input, "source.txt"), model_prefix=os.path.join(args.input, "spm"), vocab_size=32000, user_defined_symbols=['<a>', '<b>', '<brk>', '<pad>'])

with open(os.path.join(args.input, "spm.vocab"), "r") as file:
    spm_vocab = file.readlines()
    
with open(os.path.join(args.input, "dict.txt"), "w") as file:
    for line in spm_vocab:
        token = line.strip().split()[0]
        if token[0] == "<":
            file.write(f"{token} 999 #fairseq:overwrite \n")
        else:
            file.write(f"{token} 999 \n")

    