import os, json, logging
import torch
from fairseq import checkpoint_utils, data, options, tasks, utils, scoring
from fairseq.data import Dictionary, encoders
from fairseq.sequence_generator import SequenceGenerator
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu

logger = logging.getLogger()

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

if __name__ == "__main__":

    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task="no_context_tag")
    parser.add_argument('--output', default='outputs/pred.txt')
    parser.add_argument('--beam_size', default=5)
    args = options.parse_args_and_arch(parser)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task
    task = tasks.setup_task(args)
    task.load_dataset('test')
    dataset = task.datasets['test']
    task.tokenizer = dataset.dictionary.model
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collater, shuffle=False)
    dictionary = dataset.dictionary

    # Load model
    print('| loading model from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
    model = models[0]
    model.prepare_for_inference_(args)
    if use_cuda:
        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    itr = task.get_batch_iterator(
        dataset=task.dataset('test'),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions()]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    generator = task.build_generator([model], args)
    preds = []
    refs = []
    srcs = []
    #for sample in progress:
    for sample in tqdm(itr):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        src, hyps, ref = task._inference_with_bleu(generator, sample, model, True)
        srcs += src
        preds += hyps
        refs += ref

    with open(args.output, 'w') as file:
        file.write(f"BLEU score = {sacrebleu.corpus_bleu(preds, [refs]).score}\n")
        for s, h,r in zip(srcs, preds, refs):
            file.write(f"S: {s}\n")
            file.write(f"P: {h}\n")
            file.write(f"T: {r}\n")
            