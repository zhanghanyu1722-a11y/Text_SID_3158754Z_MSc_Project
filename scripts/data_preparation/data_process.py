#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MicroLens-50k -> GRID/TIGER TFRecord preprocessing
====================================================

Turn the raw MicroLens-50k files into the TFRecord layout consumed by the GRID
codebase (https://github.com/snap-research/GRID):

    <out_dir>/items/items.tfrecord.gz         one example per item: {id, text}
    <out_dir>/training/train.tfrecord.gz      user sequences  (leave-one-out: all but last 2)
    <out_dir>/evaluation/valid.tfrecord.gz    user sequences  (leave-one-out: all but last 1)
    <out_dir>/testing/test.tfrecord.gz        user sequences  (full history)

Required input files (see README section "Data acquisition"):

* --pairs   TSV with one row per user:  ``user_id <TAB> item_id item_id ...``
            (item ids must already be in chronological order, as in the
            official MicroLens ``*_pairs.tsv`` files).
* --titles  CSV with columns ``item,title`` (one row per item).

Preprocessing decisions (fixed once, reused by every configuration):

1. Item ids are remapped to the dense contiguous range ``0..N-1``, ordered by
   sorting the original item id, because GRID indexes items by dense integer ids.
   Interactions that reference items missing from the title table are dropped.
2. Leave-one-out protocol per user (only users with >= 3 interactions):
   train = all but the last two items, valid = all but the last item,
   test  = the complete history.  Users with < 3 interactions go to the
   training split only (a safeguard; never triggered in MicroLens-50k, where
   every user has >= 5 interactions).
3. Everything is serialised as gzip-compressed TFRecords.

Run from anywhere; paths may be absolute or relative to the current directory.
"""
import argparse
import os

import pandas as pd
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MicroLens-50k into GRID TFRecord files."
    )
    parser.add_argument(
        "--pairs",
        default="data/microLen/MicroLens-50k_pairs.tsv",
        help="User-interaction TSV: 'user_id<TAB>item_1 item_2 ...' (chronological).",
    )
    parser.add_argument(
        "--titles",
        default="data/microLen/MicroLens-50k_titles.csv",
        help="Item title CSV with columns 'item,title'.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/microLen",
        help="Output directory; creates items/, training/, evaluation/, testing/ under it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dir = os.path.join(args.out_dir, "training")
    eval_dir = os.path.join(args.out_dir, "evaluation")
    test_dir = os.path.join(args.out_dir, "testing")
    items_dir = os.path.join(args.out_dir, "items")
    for d in (train_dir, eval_dir, test_dir, items_dir):
        os.makedirs(d, exist_ok=True)

    print("Loading raw data and building the dense item-id mapping ...")
    df_titles = pd.read_csv(args.titles)
    # Dense remapping: original item id -> contiguous 0..N-1 (sorted by original id).
    df_titles = df_titles.sort_values("item").reset_index(drop=True)
    item_old_to_new = {
        old_id: new_id for new_id, old_id in enumerate(df_titles["item"])
    }

    # ------------------------------------------------------------------ items
    print("Writing contiguous-id items.tfrecord.gz ...")
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    items_path = os.path.join(items_dir, "items.tfrecord.gz")
    with tf.io.TFRecordWriter(items_path, options=options) as writer:
        for new_id, row in df_titles.iterrows():
            feature = {
                "id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[new_id])
                ),
                "text": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[str(row["title"]).encode("utf-8")]
                    )
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())

    # ------------------------------------------------------ interaction splits
    print("Processing interaction sequences with the leave-one-out split ...")

    def process_sequences():
        train_examples, valid_examples, test_examples = [], [], []
        with open(args.pairs, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                # Map original ids to dense ids; drop orphan ids absent from titles.
                items = [
                    item_old_to_new[int(x)]
                    for x in parts[1].split(" ")
                    if int(x) in item_old_to_new
                ]
                if len(items) < 3:
                    train_examples.append((user_id, items))
                    continue
                train_examples.append((user_id, items[:-2]))
                valid_examples.append((user_id, items[:-1]))
                test_examples.append((user_id, items))
        return train_examples, valid_examples, test_examples

    def write_seq_tfrecords(examples, output_file):
        with tf.io.TFRecordWriter(
            output_file, options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for user_id, seq in examples:
                feature = {
                    "user_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[user_id])
                    ),
                    "sequence_data": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=seq)
                    ),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

    train_data, valid_data, test_data = process_sequences()
    write_seq_tfrecords(
        train_data, os.path.join(train_dir, "train.tfrecord.gz")
    )
    write_seq_tfrecords(
        valid_data, os.path.join(eval_dir, "valid.tfrecord.gz")
    )
    write_seq_tfrecords(test_data, os.path.join(test_dir, "test.tfrecord.gz"))

    print("Done. TFRecord files written under:", args.out_dir)
    print(
        f"  items={len(df_titles)}  train={len(train_data)}  "
        f"valid={len(valid_data)}  test={len(test_data)}"
    )


if __name__ == "__main__":
    main()
