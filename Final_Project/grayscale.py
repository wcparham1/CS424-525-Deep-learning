from PIL import Image

for i in range(4001, 5001):  
    fname = 'test_set/dogs/dog.' + str(i) + '.jpg'
    img = Image.open(fname).convert('L')
    sname = 'test_set/gray_dogs/gray_scale_dog.' + str(i) + '.jpg'
    img.save(sname)