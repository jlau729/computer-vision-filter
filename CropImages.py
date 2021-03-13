from PIL import Image
import glob

'''
Censors faces in a given photo with a chosen sticker. Saves the censored version of the photo as "censored.png"
Note: If there is no faces in the given photo, then "censored.png" will be a copy of the original photo
'''

im = Image.open("uw.jpg") # image we want to censor

sticker = Image.open("pokeball.png") # sticker we want to use to censor
# size of image we potentially want to cover with sticker
cropped_width = sticker.width
cropped_height = sticker.height

# Traverses through the main image we want to censor and censors any faces with the sticker we chose
for x in range(im.width - cropped_width):
    for y in range(im.height - cropped_height):
        cropped_rectangle = (x, y, x + cropped_width, y + cropped_height)
        cropped_im = im.crop(cropped_rectangle)
        # todo: verify cropped image is a face
        isFace = False # todo change this to be result if croppsed section is face
        if isFace:
            im.paste(sticker, (x, y))
            im.save("censored.png", format = "png")