# Dialogue MT

Implementations of context-aware models for dialogue translation on fairseq.

Currently supports:

<a href="https://arxiv.org/pdf/1708.05943.pdf"> Neural Machine Translation with Extended Context</a>

* 2+1 and 2+2 concatenation models with speaker and break tags.

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6

```shell
git clone https://github.com/kayoyin/fairseq
cd fairseq
pip install --editable ./
```

## Pre-processing

A directory `DATA_PATH_DIR` should contain the files `train.json`, `valid.json`and `test.json` in the format of [WMT20 Chat Task](http://www.statmt.org/wmt20/chat-task.html).

Train the sentencepiece model:

```shell
python fairseq/learn_and_encode_spm.py --input DATA_PATH_DIR
```

This will create sentencepiece model and vocabulary in `DATA_PATH_DIR/spm.(model|vocab)` as well as a fairseq-friendly vocabulary file in `DATA_PATH_DIR/dict.txt`.

## Training

Here is an example command for training. The 2+1 and 2+2 implementations are defined by `--task {two_to_one, two_to_two}`. It should run with any model architecture.

See the [fairseq documentation](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train) for full training options.

```shell
fairseq-train DATA_PATH_DIR
--task {two_to_one, two_to_two} --arch transformer --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.2 --weight-decay 0.0 \
--max-tokens  4096  --patience 5 --seed 42 \
--save-dir checkpoints/MODEL_PATH --no-epoch-checkpoints
```

## Inference and Evaluation

For inference, run:

```shell
python fairseq/eval_dialogue_trans.py DATA_PATH_DIR \
--task {two_to_one, two_to_two} --path checkpoints/MODEL_PATH \
--beam_size 5 --output pred.txt
```

which will give model predictions for `DATA_PATH_DIR/test.json` in `pred.txt`.

If the test data is annotated,

## Datasets

|   | wmtchat2020 | openSubtitles/enfr | openSubtitles/ende | openSubtitles/enet | openSubtitles/enru |
| - | - | - | - | - | - |
| # turns |   | 41.76 (enfr) | 28.72 (ende) | 26.32 (enes) | 20.08 (enru) |
| # chats |   |   |   |   |   |

## Results

BLEU scores:

|   | wmtchat2020 | openSubtitles/enfr | openSubtitles/ende | openSubtitles/enet | openSubtitles/enru |
| - | - | - | - | - | - |
| 2+1 |   | 41.76 (enfr) | 28.72 (ende) | 26.32 (enes) | 20.08 (enru) |
| 2+2 |   |   |   |   |   |

Details for each experiment (on local):

```shell
outputs/
└── EXP_NAME.txt
logs/
└── EXP_NAME.out
scripts/
└── EXP_NAME.sh
checkpoints/
└── EXP_NAME/
    ├── checkpoint_last.pt
    └── checkpoint_best.pt
```

## Where to find my changes?

```shell
fairseq
├── data
|   └── __init__.py # register new datasets
|   └── collaters.py # custom collate()
|   └── dictionary.py # use sentencepiece tokenizer, support speaker tags
|   └── fairseq_dataset.py # add TAG_DICT
|   └── two_to_one_dataset.py 
|   └── two_to_two_dataset.py
├── tasks
|   └── two_to_one_task.py 
|   └── two_to_two_task.py 
└── utils.py # make something compatible
fairseq_cli
└──generate.py # make something compatible
```

## Todo

* Support beam search for inference
* Hyperparameters tuning (?)

