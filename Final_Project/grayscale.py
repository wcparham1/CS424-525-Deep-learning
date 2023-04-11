from PIL import Image

img = Image.open('training_set/training_set/cats/cat.1.jpg').convert('L')
img.save('gray_scale_cat.jpg')