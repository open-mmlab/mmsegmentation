import base64
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
import requests

from mmseg.apis import inference_segmentor, init_segmentor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        tmp_res = requests.post(url, image)
    with open('server_result.png', 'wb') as out_image:
        base64_str = tmp_res.content
        buffer = base64.b64decode(base64_str)
        out_image.write(buffer)
    plt.imshow(mmcv.imread('server_result.png', 'grayscale'))
    plt.show()
    model = init_segmentor(args.config, args.checkpoint, args.device)
    image = mmcv.imread(args.img)
    result = inference_segmentor(model, image)
    plt.imshow(result[0])
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
