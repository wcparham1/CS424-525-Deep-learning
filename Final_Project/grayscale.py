from PIL import Image

for i in range(1, 4001):
    fname = 'archive/training_set/cats/cat.' + str(i) + '.jpg'
    img = Image.open(fname).convert('L')
    sname = 'gray_scale_cat_'+str(i)+'.jpg'
    img.save(sname)
    
