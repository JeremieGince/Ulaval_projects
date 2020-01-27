import cv2
import numpy as np
import skimage as sk


if __name__ == '__main__':
    img_shape = (750, 1080, 3)
    labyrinth_size = (300, 320)

    background = np.zeros(img_shape)
    # labyrinth = cv2.imread('Labyrinth_img/labyrinthe002.jpg')
    labyrinth = cv2.imread("Labyrinth_img/tapis-labyrinthe.jpg")
    labyrinth = cv2.resize(labyrinth, labyrinth_size)

    img = np.array(background, dtype=np.uint8)

    top_left = background.shape[0]//2 - labyrinth.shape[0]//2
    top_right = background.shape[0]//2 + labyrinth.shape[0]//2
    bottom_left = background.shape[1]//2 - labyrinth.shape[1]//2
    bottom_right = background.shape[1]//2 + labyrinth.shape[1]//2

    labyrinth_space = img[top_left:top_right, bottom_left:bottom_right, :, ]
    print(labyrinth_space.shape, labyrinth.shape)

    img[top_left:top_right, bottom_left:bottom_right, :, ] = labyrinth

    while True:
        cv2.imshow('labyrinth', img)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
            break
