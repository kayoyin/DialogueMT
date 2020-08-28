# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    This module contains collection of classes which implement
    collate functionalities for various tasks.

    Collaters should know what data to expect for each sample
    and they should pack / collate them into batches
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import torch
from fairseq.data import data_utils


class Seq2SeqCollater(object):
    """
        Implements collate function mainly for seq2seq tasks
        This expects each sample to contain feature (src_tokens) and
        targets.
        This collator is also used for aligned training task.
    """

    def __init__(
        self,
        feature_index=0,
        label_index=1,
        pad_index=1,
        eos_index=2,
        move_eos_to_beginning=False,
    ):
        self.feature_index = feature_index
        self.label_index = label_index
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.move_eos_to_beginning = move_eos_to_beginning

    def _collate_frames(self, frames):
        """Convert a list of 2d frames into a padded 3d tensor
        Args:
            frames (list): list of 2d frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3d tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        len_max = max(frame.size(0) for frame in frames)
        f_dim = frames[0].size(1)
        res = frames[0].new(len(frames), len_max, f_dim).fill_(0.0)

        for i, v in enumerate(frames):
            res[i, : v.size(0)] = v

        return res

    def collate(self, samples):
        """
        utility function to collate samples into batch for speech recognition.
        """
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s["id"] for s in samples])
        source = data_utils.collate_tokens([s["source"] for s in samples], self.pad_index, eos_idx=self.eos_index)
        target = data_utils.collate_tokens([s["target"] for s in samples], self.pad_index, eos_idx=self.eos_index)
        
        prev_output_tokens = data_utils.collate_tokens(
                [s["target"] for s in samples],
                self.pad_index,
                self.eos_index,
                left_pad=False,
                move_eos_to_beginning=True,
            )
        # print("tgt ",target[0])
        # print("prev ",prev_output_tokens[0])
        batch = {
            "id": id,
            "ntokens": sum(len(s["target"]) for s in samples),
            "net_input": {"src_tokens": source, "src_lengths": torch.LongTensor([s.size(0) for s in source]), "prev_output_tokens":prev_output_tokens},
            "target": target,
            "nsentences": len(samples),
        }
        return batch
