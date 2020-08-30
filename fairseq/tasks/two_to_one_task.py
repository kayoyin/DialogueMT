from argparse import Namespace

import os
import torch
import logging
import json

import numpy as np
from tqdm import tqdm
from fairseq import metrics, options, utils
from fairseq.data import Dictionary, TwoToOneDataset
from fairseq.tasks import FairseqTask, register_task
import sentencepiece as spm
EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


@register_task('two_to_one')
class TwoToOneTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)


        # load dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        logger.info('[{}] dictionary: {} types'.format('Src + tgt', len(vocab)))
        vocab.model = spm.SentencePieceProcessor(model_file=os.path.join(args.data, 'spm.model'))

        return cls(args, vocab)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        path = os.path.join(self.args.data, '{}.json'.format(split))

        self.datasets[split] = TwoToOneDataset(path, self.vocab)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = self.vocab.model

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def prepare_gen(self, model, args):
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = self.vocab.model

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return (self.args.max_tokens, self.args.max_tokens)

    @property
    def source_dictionary(self):
        """Return the source and target :class:`~fairseq.data.Dictionary`."""
        return self.vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.vocab

    def _inference_with_bleu(self, generator, sample, model, return_hyps=False):
        import sacrebleu

        def decode(toks, escape_unk=False):
            toks = toks.tolist()
            #bos = task.vocab.encode("<s>")
            #eos = task.vocab.encode("</s>")
            bos = self.tokenizer.bos_id()
            eos = self.tokenizer.eos_id()
            while bos in toks:
                toks.remove(bos)
            while eos in toks:
                toks.remove(eos)
            if len(toks) == 0: 
                return ""
            s = self.tokenizer.decode(toks)
            unk_string = "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
            while "<unk>" in s:
                s.replace("<unk>", unk_string)
            return s.strip()

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.vocab.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        if return_hyps:
            return hyps, refs
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


    def eval_with_bleu(self, model, dataloader):
        import sacrebleu
        
        def decode(task, toks, escape_unk=False):
            toks = toks.tolist()
            #bos = task.vocab.encode("<s>")
            #eos = task.vocab.encode("</s>")
            bos = task.vocab.model.bos_id()
            eos = task.vocab.model.eos_id()
            while bos in toks:
                toks.remove(bos)
            while eos in toks:
                toks.remove(eos)
            s = task.vocab.decode(
                toks)
            return s.strip()

        hyps = []
        refs = []
        preds = torch.Tensor([self.vocab.model.bos_id()])
        for batch in tqdm(dataloader):
            mask_batch = batch
            mask_batch['net_input']['prev_output_tokens'] = prev_outputs
            preds = model(**mask_batch['net_input'])
            # print(preds[0][0].shape)
            # print(decode(task,torch.argmax(preds[0][0], dim=1)))
            # print(decode(task,
            #             utils.strip_pad(batch['target'][0], task.vocab.pad()),
            #             escape_unk=True,  # don't count <unk> as matches to the hypo
            #         ))
            for i in range(preds[0].shape[0]):
                hyps.append(decode(self,torch.argmax(preds[0][i], dim=1)))
                refs.append(decode(self,
                    utils.strip_pad(batch['target'][i], self.vocab.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                ))

        return sacrebleu.corpus_bleu(hyps, [refs]), hyps