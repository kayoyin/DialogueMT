import os, json
import torch
from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.data import TwoToOneDataset, Dictionary
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
def decode(task, toks, escape_unk=False):
    s = task.vocab.string(
        toks.int().cpu(),
        task.args.eval_bleu_remove_bpe,
        unk_string=(
            "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
        ),
    )
    return s

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task="no_context_tag")
parser.add_argument('--output', default='outputs/pred.txt')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
task.load_dataset('valid')
dataset = task.datasets['valid']
dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collater, shuffle=False)


# Load model
print('| loading model from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
model = models[0]
hyps = []
refs = []
# for batch in tqdm(dataloader):
#     preds = model(**batch['net_input'])
#     # print(preds[0][0].shape)
#     # print(decode(task,torch.argmax(preds[0][0], dim=1)))
#     # print(decode(task,
#     #             utils.strip_pad(batch['target'][0], task.vocab.pad()),
#     #             escape_unk=True,  # don't count <unk> as matches to the hypo
#     #         ))
#     for i in range(preds[0].shape[0]):
#             hyps.append(decode(task,torch.argmax(preds[0][i], dim=1)))
#             refs.append(decode(task,
#                 utils.strip_pad(batch['target'][i], task.vocab.pad()),
#                 escape_unk=True,  # don't count <unk> as matches to the hypo
#             ))


# iterator = task.get_batch_iterator(task.datasets['valid'], args.max_tokens)
# gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
# task.sequence_generator = task.build_generator([model], args)

# hyp = []
# ref = []
# for sample in iterator._get_iterator_for_epoch(iterator.epoch, False):
#     # Feed batch to the model and get predictions
#     hyps, refs = task._inference_with_bleu(task.sequence_generator, sample, model)
#     hyp += hyps
#     ref += refs

bleu, hyps = task.eval_with_bleu(model, dataloader)
print(bleu.score)
with open(args.output, 'w') as file:
    file.write(f"BLEU score = {bleu.score}\n")
    for h in hyps:
        file.write(h + "\n")
