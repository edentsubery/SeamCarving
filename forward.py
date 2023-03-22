def forward(im):
    row, columns = im.shape[:2]
    im = to_grayscale(im)
    energy = np.zeros((row, columns))
    M = np.zeros((row, columns))
    V = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    Cv = np.abs(R - L)
    Cl = np.abs(V - L) + cV
    Cr = np.abs(V - R) + cV
    for i in range(1, row):
        Mv = M[i-1]
        Ml = np.roll(Mv, 1)
        Mr = np.roll(Mv, -1)
        arrM = np.array([Mv, Ml, Mr])
        arrC = np.array([Cv[i], Cl[i], Cr[i]])
        arrM += arrC
        minSeam = np.argmin(arrM, axis=0)
        M[i] = np.choose(minSeam, arrM)
        energy[i] = np.choose(minSeam, arrC)        
    return energy