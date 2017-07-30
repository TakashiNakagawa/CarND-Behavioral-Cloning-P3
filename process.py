import cv2

image = cv2.imread("./images/center_2017_07_18_21_11_01_154.jpg")

flip = cv2.flip(image,1)

crop = image[70:135, :]

cv2.imwrite("./flip.jpg", flip)
cv2.imwrite("./crop.jpg", crop)
