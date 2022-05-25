from PIL import Image, ImageEnhance
import os


def processing(_filename: str):
    im = Image.open(_filename)

    _filename = _filename[0: len(_filename) - 4]

    enhancer = ImageEnhance.Brightness(im)

    # Rotate the original image of 90, 180, and 270°
    for i in [90, 180, 270]:
        try:
            im_output = im.rotate(i)
            im_output.save(_filename + "_" + str(i) + ".jpg")
        except Exception:
            print(_filename + " ignored")

    # Rotate the darkened image of 0, 90, 180, and 270°
    factor = 0.6
    try:
        im_output = enhancer.enhance(factor)
        for i in [0, 90, 180, 270]:
            im_output = im_output.rotate(i)
            im_output.save(_filename + "_dark_" + str(i) + ".jpg")
    except Exception:
        print(_filename + " ignored")

    # Rotate the brightened image of 0, 90, 180, and 270°
    factor = 1.4
    try:
        im_output = enhancer.enhance(factor)
        for i in [0, 90, 180, 270]:
            im_output = im_output.rotate(i)
            im_output.save(_filename + "_bright_" + str(i) + ".jpg")
    except Exception:
        print(_filename + " ignored")
        

def database_processing(_db):
    for rep in os.listdir('../databases/' + _db):
        for fruit in os.listdir('../databases/' + _db + '/' + rep):
            for file in os.listdir('../databases/' + _db + '/' + rep + '/' + fruit):
                processing('../databases/' + _db + '/' + rep + '/' + fruit + '/' + file)
