import os, json, logging
import torch
from fairseq import checkpoint_utils, data, options, tasks, utils, scoring
from fairseq.data import TwoToOneDataset, Dictionary, encoders
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
        max_sentences=2,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
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

    generator = task.build_generator(models, args)
    preds = []
    refs = []
    #for sample in progress:
    for sample in tqdm(itr):
        has_target = sample['target'] is not None
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]
        
        hypos = task.inference_step(generator, models, sample, prefix_tokens=prefix_tokens, constraints=constraints)
        
        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            if 'src_tokens' in sample['net_input']:
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], dictionary.pad())
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], dictionary.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                refs.append(target_str)
            else:
                src_str = dataset.src[sample_id] # TODO: include previous sentence here for alignment??
                if has_target:
                    target_str = dataset.tgt[sample_id]
                    refs.append(target_str)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=dictionary,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = hypo_str
                preds.append(hypo_str)
    with open(args.output, 'w') as file:
        if has_target:
            file.write(f"BLEU score = {sacrebleu.corpus_bleu(preds, [refs]).score}\n")
        for h in preds:
            file.write(f"{h}\n")
