from PIL import Image

<<<<<<< HEAD
for i in range(4001, 5001):  
    fname = 'test_set/dogs/dog.' + str(i) + '.jpg'
    img = Image.open(fname).convert('L')
    sname = 'test_set/gray_dogs/gray_scale_dog.' + str(i) + '.jpg'
    img.save(sname)
=======
for i in range(1, 4001):
    fname = 'archive/training_set/cats/cat.' + str(i) + '.jpg'
    img = Image.open(fname).convert('L')
    sname = 'gray_scale_cat_'+str(i)+'.jpg'
    img.save(sname)
    
>>>>>>> e544eed3e7d24becd244d43940cd65cdfb0271d1
