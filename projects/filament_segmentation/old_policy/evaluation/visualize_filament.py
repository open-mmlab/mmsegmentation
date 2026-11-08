"""
フィラメント検出結果の可視化スクリプト

赤  : FilFinderラベルのみ（FN: 見逃し）
シアン: 推定結果のみ（FP: 誤検出）
白  : ラベルと推定結果の一致（TP: 正解）
"""

import os
import argparse
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm


def overlay(input_path, label_path, pred_path, output_path):
    inp = np.array(Image.open(input_path).convert('L'))
    vis = np.stack([inp, inp, inp], axis=2)
    label = np.array(Image.open(label_path))
    pred  = np.array(Image.open(pred_path))

    tp = (label == 1) & (pred == 1)
    fn = (label == 1) & (pred == 0)
    fp = (label == 0) & (pred == 1)

    vis[tp] = [255, 255, 255]
    vis[fn] = [255,  80,  80]
    vis[fp] = [ 80, 220, 220]

    Image.fromarray(vis.astype(np.uint8)).save(output_path)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.png')))

    if len(pred_files) == 0:
        print(f"ERROR: 予測ファイルが見つかりません: {args.pred_dir}")
        return

    print(f"対象ファイル数: {len(pred_files)}")

    for pred_path in tqdm(pred_files):
        basename = os.path.basename(pred_path)
        # predのファイル名: HGBS_orionB_..._in.png
        stem = basename.replace('_in.png', '')

        # input/labelはtilesサブディレクトリ以下を検索
        input_files = glob.glob(os.path.join(args.input_dir, '**', stem + '_in.png'), recursive=True)
        label_files = glob.glob(os.path.join(args.label_dir, '**', stem + '_out.png'), recursive=True)

        if not input_files:
            print(f"SKIP (input not found): {stem}")
            continue
        if not label_files:
            print(f"SKIP (label not found): {stem}")
            continue

        out_path = os.path.join(args.output_dir, stem + '_vis.png')
        overlay(input_files[0], label_files[0], pred_path, out_path)

    print(f"完了: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    main(args)
