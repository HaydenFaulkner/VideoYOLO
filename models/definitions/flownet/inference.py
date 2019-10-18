import cv2
import mxnet as mx
import numpy as np
from scipy.misc import imresize
from tqdm import tqdm

from flownet import get_flownet
from utils import flow_to_image, crop, normalise

def process_two_images(model, imgs, ctx=None):
    """
    Process two images into one flow image
    Args:
        model: The model to use
        imgs: a list of 2 images
        ctx: the model ctx

    Returns:

    """
    if len(imgs) != 2:
        return None
    if isinstance(imgs[0], str):
        if os.path.exists(imgs[0]):
            imgs[0] = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB)
        else:
            return None
    if isinstance(imgs[1], str):
        if os.path.exists(imgs[1]):
            imgs[1] = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB)
        else:
            return None

    imgs = crop(imgs)
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = normalise(imgs)

    imgs = mx.nd.array(imgs, ctx=ctx)
    imgs = mx.nd.expand_dims(imgs, 0)  # add batch axis

    flow = model(imgs)  # run the model

    flow = flow.asnumpy()
    flow = flow.squeeze()
    flow = flow.transpose(1, 2, 0)
    img = flow_to_image(flow)
    img = imresize(img, 4.0)  # doing the bilinear interpolation on the img, NOT flow cause was too hard :'(

    return img, flow


def process_imagedir(model, input_dir, output_dir=None, ctx=None):
    """
    Process a directory of images

    Args:
        model:
        input_dir:
        output_dir:
        ctx:

    Returns:

    """

    files = []
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(input_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    if not len(files) > 0:
        print("Couldn't find any files in {}".format(input_dir))
        return None

    files.sort()

    for i in tqdm(range(len(files) - 1), desc='Calculating Flow'):
        img, flow = process_two_images(model, files[i:i+2], ctx)
        dir, file = os.path.split(files[i])
        if output_dir is None:
            output_dir = os.path.join(dir, 'flow')
        os.makedirs(output_dir, exists_ok=True)
        cv2.imwrite(os.path.join(output_dir, file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return output_dir


def process_video(model, input_path, output_path=None, ctx=None):
    """
    Process a video into a flow video

    Args:
        model:
        input_path:
        output_path:
        ctx:

    Returns:

    """
    capture = cv2.VideoCapture(input_path)
    frames = []
    while_safety = 0
    while len(frames) < 200:# int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1:
        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        if image is None:
            while_safety += 1
            continue

        while_safety = 0  # reset the safety count
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()

    if len(frames) < 2:
        return None

    if output_path is None:
        output_path = input_path[:-4] + '_flow.mp4'

    cropped_frames = crop(frames)
    h, w, _= cropped_frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (w, h))

    for i in tqdm(range(len(frames)-1), desc='Calculating Flow'):
        mx.nd.waitall()
        img, flow = process_two_images(model, frames[i:i+2], ctx)
        video.write(img)

    video.release()  # release the video

    return output_path

if __name__ == '__main__':
    # just for debugging

    # save_path = "models/definitions/flownet/weights/FlowNet2-S_checkpoint.params"
    save_path = "models/definitions/flownet/weights/FlowNet2-C_checkpoint.params"

    ctx = mx.gpu(0)

    # net = get_flownet('S', pretrained=True, ctx=ctx)
    net = get_flownet('C', pretrained=True, ctx=ctx)
    net.hybridize()

    input_path = "/path/to/test.mp4"
    process_video(net, input_path, ctx=ctx)

    print("DONE")
