def dinamicProgramming(img):
    row, column, _ = img.shape
    #first we want to know the energy of each pixel in the image
    energys = EnereyForImg(img)
    #we will work on copy of this picture
    M = energys.copy()
    init = np.zeros_like(M, dtype=np.int)
    for i in range(1, row):
        for j in range(0, column):
            if j == 0:
                index = np.argmin(M[i - 1, j:j + 2])
                init[i, j] = index + j
                M[i, j] += M[i - 1, index + j]
            else:
                index = np.argmin(M[i - 1, j - 1:j + 2])
                init[i, j] = index + j - 1
                M[i, j] += M[i - 1, index + j - 1]
    return M, init