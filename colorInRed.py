#function to color the verticl seams with red
def redColor(seams, img):
    seamCarving = np.copy(img) 
    x, y = np.transpose([(i,int(j)) for i,j in enumerate(seams)])
    seamCarving[x, y] = (0,0,255)
    return seamCarving