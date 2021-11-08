from argparse import ArgumentParser
from io import BytesIO

import matplotlib.pyplot as plt
import mmcv
import requests

from mmseg.apis import inference_segmentor, init_segmentor


def parse_args():
    parser = ArgumentParser(
        description='Compare result of torchserve and pytorch,'
        'and visualize them.')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--result-image',
        type=str,
        default=None,
        help='save server output in result-image')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()
    return args


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        tmp_res = requests.post(url, image)
    content = tmp_res.content
    if args.result_image:
        with open(args.result_image, 'wb') as out_image:
            out_image.write(content)
        plt.imshow(mmcv.imread(args.result_image, 'grayscale'))
        plt.show()
    else:
        plt.imshow(plt.imread(BytesIO(content)))
        plt.show()
    model = init_segmentor(args.config, args.checkpoint, args.device)
    image = mmcv.imread(args.img)
    result = inference_segmentor(model, image)
    plt.imshow(result[0])
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
