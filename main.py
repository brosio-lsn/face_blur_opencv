import cv2 as cv
import numpy as np

# fetching cascade files for front face and profile face
front_face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
profile_face_cascade = cv.CascadeClassifier("haarcascade_profileface.xml")
# fetching the image to analyse and copying it
img = cv.imread('faces.jpg')
copy = img.copy()

# turn to grayscale to analyse
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# rectangles list to store all the rectangles (useful to make sure that the different analyses don't put a rectangle
# on the same faces many times) the process of making sure the rectangles don't overlap for the same face isn't that
# useful when only wanting to blur the face, but it is useful for face detection in general(more of a personal
# challenge)
rectangles = []

# function to determine the area shared by 2 rectangles
def common_area(rectangleA, rectangleB):
    x1A, y1A, w, h = rectangleA.reshape(4)
    y2A = y1A + h
    x2A = x1A + w
    x1B, y1B, w, h = rectangleB.reshape(4)
    y2B = y1B + h
    x2B = x1B + w
    # checking that the x coordinates of the 2 rectangles intersect
    if not ((x1A <= x1B <= x2A) or (x1B <= x1A <= x2B)):
        return 0
    if not ((y1A <= y1B <= y2A) or (y1B <= y1A <= y2B)):
        return 0
    # width of the common area
    l = min(x2A, x2B) - max(x1A, x1B)
    # height of the common area
    L = min(y2A, y2B) - max(y1A, y1B)
    return L * l


# function to display the rectnagles for the faces and make sure they don't overlap with the previous ones
def check_and_draw_rectangle(rectangle):
    x, y, w, h = rectangle.reshape(4)
    # check if the rectangle isn t too close in another one already in the list and display it and add it to list if not
    already_a_similar_rectangle = False
    for rec in rectangles:
        # checking the area of the new rectangle isn't 0 (to prevent division by 0 in the calculus below)
        if (w * h == 0):
            break;
        area = common_area(rectangle, rec)
        area_percentage_rec = area / (w * h)
        area_percentage_rectangle = area / (rec[2] * rec[3])
        # the criteria to check if a similar rectangle is already here is if the new rectangle would share more than 70% of its area
        # with a rectangle already in the list
        if ((area_percentage_rectangle > 0.7) or (area_percentage_rec > 0.7)):
            already_a_similar_rectangle = True
            break
    if (not already_a_similar_rectangle):
        rectangles.append(rectangle)


# function to analyse the given image with the given cascade adn draw the result
def analse_image_with_cascade(cascade, image):
    faces = cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=3)
    for rectangle in faces:
        check_and_draw_rectangle(rectangle)


# analyse front face
analse_image_with_cascade(front_face_cascade, gray)

# analyse one side of the face (the left side)
analse_image_with_cascade(profile_face_cascade, gray)

# analyse for the other side of the face(the right side)
    # reversing the image to run the face analyse for the lest side (because we have only left side cascade)
reverse_image = gray.copy()
for i in range(0, len(gray)):
    for j in range(0, len(gray[0])):
        reverse_image[i][j] = gray[i][len(gray[0]) - 1 - j]
    # applying the profile analyses on the reversed image
faces = profile_face_cascade.detectMultiScale(reverse_image, scaleFactor=1.2, minNeighbors=3)
    # rereversing the rectangles found to make them adapted to the unreversed image, then drawing them if needed
for rectangle in faces:
    x, y, w, h = rectangle.reshape(4)
    check_and_draw_rectangle(np.array([len(gray) - x - w, y, w, h]))

# now we can blur what is inside the rectangles
if (len(rectangles) != 0):
    blured = cv.GaussianBlur(img, (7, 7), 50)
    # for each rectangle, we replace each pixel of the result image by the pixel at the same location of the blured
    # original image
    for rectangle in rectangles:
        x, y, w, h = rectangle.reshape(4)
        for i in range(y, y + h):
            for j in range(x, x + w):
                copy[i][j] = blured[i][j]

# displaying the result
cv.imshow('faces', copy)
cv.waitKey(0)
