#Deleting the pixels from the seam with the least energy
def deleteSmallestSeams(img):
    row, column, _ = img.shape
    M, init = minSeam(img)
    goingToDelete = np.ones((row, column), dtype=np.bool)
    #The first selected cell at the bottom row, is the one with the minimal value.
    j = np.argmin(M[-1])
    #pass all the pixles and choose the seams we need to remove
    for i in reversed(range(row)):
        goingToDelete[i, j] = False
        j = init[i, j]
    goingToDelete = np.stack([goingToDelete] * 3, axis=2)
    img = img[mask].reshape((row, column-1, 3))
    return img

#iterations for all columns
def iteration(img):
    row, column, _ = img.shape
    for i in range(column - int(0.5*column)): 
        img = deleteSmallestSeams(img)
    return img