import cv2
import numpy as np
import tifffile
import os
import imutils

# data stretching fuction
def bytescaling(data, cmin=None, cmax=None, high=255, low=0):

    if data.dtype == np.uint8:
        return data
    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin

    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

# numpy를 preprocessing하여 png로 바꿔주는 함수
def make_numpy_to_png(_numpy, save_file_name):
    # _numpy.shape ==> (W, H, 4)
    # type(_numpy) ==> <class 'numpy.ndarray'>
    # print(save_file_name) ==> './sample.tif'

    # (W, H, 4) ==> (W, H, 3)
    img = cv2.cvtColor(_numpy, cv2.IMREAD_COLOR)

    # data stretching
    img = bytescaling(img)

    # CLAHE preprocessing
    b, g ,r = cv2.split(img)
    cla = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    b = cla.apply(b)
    g = cla.apply(g)
    r = cla.apply(r)
    img = cv2.merge((b, g, r))

    # save png file to save_file_name
    # img.shape ==> (W, H, 3)
    # type(img) ==> <class 'numpy.ndarray'>
    cv2.imwrite(save_file_name, img)


def sliding_window(image, stepSize, windowSize, out_path, img_name):

    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            tmp = image[y:y + windowSize[1], x:x + windowSize[0]]
            save_file_name = os.path.join(out_path, img_name.split('.')[0] + '_' + str(x) + '_' + str(y) + '.png')
            make_numpy_to_png(tmp, save_file_name)


if __name__ == "__main__":
    
    main_path = './whole_test'
    out_path = './out'
    path_list = [x for x in os.listdir(main_path)]

    for _, img_name in enumerate(path_list):
        cur_path = os.path.join(main_path, img_name)

        img = tifffile.imread(cur_path)

        sliding_window(img, 224, (224, 224), out_path, img_name)