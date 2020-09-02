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


|  | wmtchat2020 | openSubtitles/enfr | openSubtitles/ende | openSubtitles/enet | openSubtitles/enru |
| - | - | - | - | - | - |
| Turns | 13845/15747/17847 | 246540/251458/259276 | 109241/111316/114633 | 174218/177376/182527 | 157880/161063/165949 |
| Chats | 550/628/706 | 6997/7137/7346 | 3582/3652/3760 | 4394/4482/4614 | 23126/23588/24282 |
| English turns | 7629/8669/9802 | 139849/142797/147637 | 55539/56572/58287 | 130598/133152/137362 | 157880/161063/165949 |
| Other turns | 6216/7078/8045 | 106691/108661/111639 | 53702/54744/56346 | 43620/44224/45165 | 133636/136346/140486 |

## Results

BLEU scores:


|    | openSubtitles/enfr | openSubtitles/ende | openSubtitles/enet | openSubtitles/enru |
| - | - | - | - | - |
| Context-agnostic (Kayo)| 43.55(enfr_k)|31.67 (ende0)| | |
| Context-agnostic (Kervy)| 46.56|30.90|27.38 | 21.49 |
| 2+1 |   42.24 (enfr_2) | 28.72 (ende) | 26.48 (enet_2) | 20.20 (enru_2) |
| 2+2 |   43.90 (enfr2_3)  |  31.07 (ende2) | 27.06 (enet2)  | 21.38 (enru2)  |

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
