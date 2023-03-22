import numpy as np
import utils

USE_FORWARD_IMPLEMENTATION=False


def backward_energy_matrix(img):
    return utils.get_gradients(img)


def forward_looking_energy_matrix(img): #gray_image
    height, width = img.shape
    energy = np.zeros((height, width))
    m = np.zeros((height, width))

    U = np.roll(img, 1, axis=0)
    L = np.roll(img, 1, axis=1)
    R = np.roll(img, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, height):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    return energy


def remove_seams_from_image(img, boolmask,k):
    h, w = img.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return img[boolmask3c].reshape((h, w - k, 3))


def remove_k_seams_from_image(img, seams,k):
    height, width=img.shape[:2]
    seams_transpose=np.transpose(seams)
    update_width = width-1
    resized = np.zeros((update_width, height))
    for i in range(height):
        #for j in range(width):
        # build bool mask
        mask = [i*j+j for j in range(width)]
        boolmask = np.isin(seams_transpose[i], mask)
        for m in range(len(boolmask)-2):
            if(i<261 and m<261):
                print("i and m: ", i, m)
                resized[i][m] = boolmask[m]


    #resized = remove_seams_from_image(img,boolmask,k)
    #return resized
    return boolmask


def remove_seam_from_matrix(matrix, seam):
    height, width = matrix.shape
    del_ind = seam
    mask = np.ones((height, width), dtype=bool)
    mask[range(height), del_ind] = False
    return matrix[mask].reshape(height, width - 1)


def find_minimal_seam(gray_img):
    height, width = gray_img.shape
    print(height, width)
    energy_function = forward_looking_energy_matrix if USE_FORWARD_IMPLEMENTATION else backward_energy_matrix
    M = energy_function(gray_img)
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, height):
        for j in range(0, width):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    seam_idx = []
    j = np.argmin(M[-1])
    for i in range(height - 1, -1, -1):
        seam_idx.append(j)
        j = backtrack[i, j]

    return seam_idx[::-1]


def calculate_original_indices_seam(seam_in_resized, index_matrix):
    original_indices_seam=[]
    rows= len(seam_in_resized)
    for i in range(rows):
        original_indices_seam.append(index_matrix[i][seam_in_resized[i]])
    return original_indices_seam


def remove_k_seams(gray_img, k):
    row, column = gray_img.shape
    repetitions = (row, 1)
    index_matrix = np.tile(np.arange(column), repetitions)
    actual_seams = []
    seams=[]
    temp_img=gray_img.copy()
    for i in range(k):
        seam_in_resized=find_minimal_seam(temp_img)
        seam=calculate_original_indices_seam(seam_in_resized, index_matrix)
        actual_seams.append(seam)
        seams.append(seam_in_resized)
        print(seam_in_resized)
        temp_img = remove_seam_from_matrix(temp_img, seam_in_resized)
        index_matrix = remove_seam_from_matrix(index_matrix, seam_in_resized)
    actual_seams.reverse()
    seams.reverse()
    return index_matrix, actual_seams, seams


def add_seam(image, seam):
    height, width = image.shape[:2]
    resized = np.zeros((height, width + 1, 3))
    for row in range(height):
        col = seam[row]
        for ch in range(3):
            if col == 0:
                p = np.average(image[row, col: col + 2, ch])
                resized[row, col, ch] = image[row, col, ch]
                resized[row, col + 1, ch] = p
                resized[row, col + 1:, ch] = image[row, col:, ch]
            else:
                p = np.average(image[row, col - 1: col + 1, ch])
                resized[row, : col, ch] = image[row, : col, ch]
                resized[row, col, ch] = p
                resized[row, col + 1:, ch] = image[row, col:, ch]

    return resized


def duplicate_k_seams_from_image(image, seams,boolmasks):

    for _ in range(len(seams)):
        boolmask=boolmasks.pop()
        seam = seams.pop()
        image = add_seam(image, boolmask)

        for remaining_seam in seams:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return image


def seam_carv(image, gray_image, hoizontal, k, forward_implementation):
    size_up = k <= 0
    size_down = k >= 0
    # if the flag of horizontal - we want to do it like vertical and just rotate the image
    if hoizontal:
        image = np.rot90(image, 1)
        gray_image = np.rot90(gray_image, 1)
    index_matrix, actual_seams,seams = remove_k_seams(gray_image, abs(k))
    print(seams)
    if size_up:
        resized = duplicate_k_seams_from_image(image,seams,k)
    if size_down:
        resized = remove_k_seams_from_image(image, seams,k)
    if hoizontal:
        colored = Color(seams, image, index_matrix, True)
        np.rot90(resized, 3,axes=(0, 1))
        np.rot90(colored, 3, axes=(0, 1))
    else:
        colored = Color(seams, image, index_matrix, False)
    return resized, colored, index_matrix


# function to get the horizontal/seams seams in black/red:
def Color(seams, img, init, black):
    colored = np.copy(img)
    row, col, _ = img.shape
    print("in color:")
    for i in range(len(seams)):
        if(seams[i][0]<262):
            if (black):
                colored[seams[i][0]][seams[i][1]] = 0
            else:
                colored[seams[i][0]][seams[i][1]] = 108
    return colored


# main function for seam_carving:
def resize(image, height_req, width_req, forward_implementation):
    # first take a copy of the image in gray color
    img_gray = utils.to_grayscale(image)
    # data get an matrix that represent the gray image
    data = np.asarray(img_gray)
    height, width = data.shape
    # calc 3 matrix - the resize one, the red one and the black one
    img_in_red, img_in_black, img = image, image, image
    if width != width_req:
        img, img_in_red, init = seam_carv(image, data, False, width - width_req, forward_implementation)
    if height != height_req:
        img, img_in_black, init = seam_carv(image, data, True, height - height_req, forward_implementation)
    return {'resized': img, 'black': img_in_black, 'red': img_in_red}
