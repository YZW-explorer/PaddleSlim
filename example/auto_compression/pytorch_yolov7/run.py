# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression
from dataset import COCOValDataset, COCOTrainDataset
from post_process import YOLOv7PostProcess, coco_metric


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    parser.add_argument(
        '--eval', type=bool, default=False, help="whether to run evaluation.")

    return parser


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for data in val_loader:
            data_all = {k: np.array(v) for k, v in data.items()}
            outs = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: data_all['image']},
                           fetch_list=test_fetch_list,
                           return_numpy=False)
            res = {}
            postprocess = YOLOv7PostProcess(
                score_threshold=0.001, nms_threshold=0.65, multi_label=True)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
            bboxes_list.append(res['bbox'])
            bbox_nums_list.append(res['bbox_num'])
            image_id_list.append(np.array(data_all['im_id']))
            t.update()
    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)
    return map_res[0]


def main():
    global global_config
    all_config = load_slim_config(FLAGS.config_path)
    assert "Global" in all_config, "Key 'Global' not found in config file. \n{}".format(
        all_config)
    global_config = all_config["Global"]

    dataset = COCOTrainDataset(
        dataset_dir=global_config['dataset_dir'],
        image_dir=global_config['train_image_dir'],
        anno_path=global_config['train_anno_path'])
    train_loader = paddle.io.DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    if 'Evaluation' in global_config.keys() and global_config[
            'Evaluation'] and paddle.distributed.get_rank() == 0:
        eval_func = eval_function
        global val_loader
        dataset = COCOValDataset(
            dataset_dir=global_config['dataset_dir'],
            image_dir=global_config['val_image_dir'],
            anno_path=global_config['val_anno_path'])
        global anno_file
        anno_file = dataset.ann_file
        val_loader = paddle.io.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0)
    else:
        eval_func = None

    ac = AutoCompression(
        model_dir=global_config["model_dir"],
        train_dataloader=train_loader,
        save_dir=FLAGS.save_dir,
        config=all_config,
        eval_callback=eval_func)
    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()