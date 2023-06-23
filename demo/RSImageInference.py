from argparse import ArgumentParser
from mmseg.apis import inference_model, init_model
import numpy as np
import os
from osgeo import gdal
import queue
import threading


class RSImage:
    def __init__(self, img):
        if isinstance(img, str):
            self.dataset = gdal.Open(img)
            self.path = img
        else:
            self.dataset = img
        if self.dataset is not None:
            self.width = self.dataset.RasterXSize
            self.height = self.dataset.RasterYSize
            self.channel = self.dataset.RasterCount
            self.trans = self.dataset.GetGeoTransform()
            self.proj = self.dataset.GetProjection()
            self.bandlist = []
            for c in range(self.channel):
                self.bandlist.append(self.dataset.GetRasterBand(c + 1))
            self.grids = []
        else:
            raise Exception(f'{img} is not a image')

    def imread(self, grid=None):
        if grid is not None:
            return np.einsum('ijk->jki', self.dataset.ReadAsArray(*grid[0:4]))
        else:
            return np.einsum('ijk->jki', self.dataset.ReadAsArray())

    def imwrite(self, grid=None, data=None):
        if grid is not None:
            for index, band in enumerate(self.bandlist):
                band.WriteArray(data[grid[5]:grid[5] + grid[7], grid[4]:grid[4] + grid[6]],
                                grid[0] + grid[4], grid[1] + grid[5])
        else:
            if data is not None:
                for i in range(self.channel):
                    self.bandlist[i].WriteArray(data[..., i])

    def create_segmap(self, segmap_path=None):
        if segmap_path is None:
            base_name = os.path.basename(self.path)
            file_name, _ = os.path.splitext(base_name)
            segmap_path = f"{file_name}_label.tif"
        driver = gdal.GetDriverByName('GTiff')
        segmap = driver.Create(segmap_path, self.width, self.height, 1, gdal.GDT_Byte)
        segmap.SetGeoTransform(self.trans)
        segmap.SetProjection(self.proj)
        segmap_img = RSImage(segmap)
        segmap_img.path = segmap_path
        return segmap_img

    def create_grids(self, winx, winy, xstep=0, ystep=0):
        if xstep == 0:
            xstep = winx
        if ystep == 0:
            ystep = winy

        x_half_overlap = (winx - xstep + 1) // 2
        y_half_overlap = (winy - ystep + 1) // 2

        y = 0
        x = 0

        while y < self.height:
            if y + winy >= self.height:
                yoff = self.height - winy
                ysize = winy
                y_end = True
            else:
                yoff = y
                ysize = winy
                y_end = False
            if yoff == 0:
                y_crop_off = 0
                y_crop_size = winy
            else:
                y_crop_off = y_half_overlap
                y_crop_size = winy - y_half_overlap
            if not y_end:
                y_crop_size -= y_half_overlap
            while x < self.width:
                if x + winx >= self.width:
                    xoff = self.width - winx
                    xsize = winx
                    x_end = True
                else:
                    xoff = x
                    xsize = winx
                    x_end = False
                if xoff == 0:
                    x_crop_off = 0
                    x_crop_size = winx
                else:
                    x_crop_off = x_half_overlap
                    x_crop_size = winx - x_half_overlap
                if not x_end:
                    x_crop_size -= x_half_overlap
                self.grids.append([xoff, yoff, xsize, ysize, x_crop_off, y_crop_off, x_crop_size, y_crop_size])
                x += xstep
                if x_end:
                    break
            y += ystep
            x = 0
            if y_end:
                break


def reader(img: RSImage, read_buffer, end_flag, xsize, ysize, xstep, ystep):
    img.create_grids(xsize, ysize, xstep, ystep)
    for grid in img.grids:
        read_buffer.put([grid, img.imread(grid=grid)])
    read_buffer.put(end_flag)


def inferencer(read_buffer, write_buffer, end_flag, model):
    while True:
        item = read_buffer.get()
        if item == end_flag:
            read_buffer.put(end_flag)
            write_buffer.put(item)
            break
        result = inference_model(model, item[1])
        item[1] = result.pred_sem_seg.cpu().data.numpy()[0]
        write_buffer.put(item)
        read_buffer.task_done()


def writer(img: RSImage, write_buffer, end_flag, segmap_path):
    segmap = img.create_segmap(segmap_path)
    while True:
        item = write_buffer.get()
        if item == end_flag:
            break
        segmap.imwrite(grid=item[0], data=item[1])
        write_buffer.task_done()


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out', help='Path to save result image')
    parser.add_argument('batch_size', type=int,
                        help='maximum number of windows inferred simultaneously')
    parser.add_argument('win_size', help='window xsize,ysize', type=int, nargs=2)
    parser.add_argument('stride', help='window xstride,ystride', type=int, nargs=2)
    parser.add_argument('--thread', default=1, type=int,
                        help='number of inference threads')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference')
    args = parser.parse_args()

    img = RSImage(args.img)
    model = init_model(args.model, args.checkpoint, device=args.device)
    read_buffer = queue.Queue(args.batch_size)
    write_buffer = queue.Queue(args.batch_size)
    end_flag = object()

    read_thread = threading.Thread(target=reader,
                                   args=(img,
                                         read_buffer,
                                         end_flag,
                                         *args.win_size,
                                         *args.stride))
    read_thread.start()
    inferencer_threads = []
    for _ in range(args.thread):
        inferencer_thread = threading.Thread(target=inferencer,
                                             args=(read_buffer,
                                                   write_buffer,
                                                   end_flag,
                                                   model))
        inferencer_thread.start()
        inferencer_threads.append(inferencer_thread)
    write_thread = threading.Thread(target=writer,
                                    args=(img,
                                          write_buffer,
                                          end_flag,
                                          args.out))
    write_thread.start()
    read_thread.join()
    for inferencer_thread in inferencer_threads:
        inferencer_thread.join()
    write_thread.join()


if __name__ == '__main__':
    main()
