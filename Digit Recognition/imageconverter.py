from PIL import Image
import scipy.io

img = Image.open('6greyscale20x20.jpg').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
#data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].

# For example:
#for row in data:
 #   print(' '.join('{:3}'.format(value) for value in row))
scipy.io.savemat('customdata.mat', dict(x=data))
