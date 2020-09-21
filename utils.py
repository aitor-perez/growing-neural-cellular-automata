import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import os
import glob


def clear_img_folder(which):
    files = glob.glob("img/" + which)
    for f in files:
        os.remove(f)


def read(file):
    img = Image.open(file)
    tens = torch.from_numpy(np.array(img)).type(torch.float)
    return tens / 255


def to_nwhc(ncwh):
    return ncwh.permute(0, 2, 3, 1)


def to_ncwh(nwhc):
    return nwhc.permute(0, 3, 1, 2)


def rechannel_nwhc(nwhc, c2):
    n = nwhc.shape[0]
    w = nwhc.shape[1]
    h = nwhc.shape[2]
    c1 = nwhc.shape[3]

    if c2 <= c1:
        return nwhc[:, :, :, 0:c2]
    else:
        tens = torch.zeros((n, w, h, c2))
        tens[:, :, :, 0:c1] = nwhc
        return tens


def old_display(tens, instance, iteration):
    tens = to_nwhc(tens)
    tens = rechannel_nwhc(tens, 4)

    n = tens.shape[0]

    arr = np.uint8(tens[instance, :, :, :].detach().numpy() * 255)

    plt.rcParams['savefig.facecolor'] = 'white'
    plt.imsave("./img/" + str(instance) + "-" + str(iteration) + ".png", arr)
    plt.show()


def make_anim(iterations, instance, epoch):
    images = []
    for i in range(iterations):
        img = Image.open("img/" + str(instance) + "-" + str(i) + ".png").convert("RGBA")
        img_data = img.getdata()

        # Set transparent pixels to white for better visualization
        white_bg_img_data = []
        for d in img_data:
            if d[3] == 0:
                white_bg_img_data.append((255, 255, 255, 1))
            else:
                white_bg_img_data.append(d)

        img.putdata(white_bg_img_data)
        images.append(img)

    filename = "img/ins-" + str(instance) + "-epoch-" + str(epoch) + ".gif"
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)


