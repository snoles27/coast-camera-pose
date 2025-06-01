import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('test_photos/w3_youtube_0838_1032_edward_island.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)

contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 4. Iterate through contours and extract points (e.g., by drawing)
for contour in contours:
    # Approximate the contour to reduce points (optional)
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    # Iterate through points (approximated contour points)
    for point in approx:
        # Access the coordinates of the point
        x, y = point[0]  # point[0] is a list, access the first element for x and y
        
        # Draw the points (optional)
        cv.circle(img, (x, y), 3, (0, 255, 0), -1) # Green circle

# Display the result
cv.imshow('Original Image with Points', img)
cv.waitKey(0)
cv.destroyAllWindows()

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
# plt.show()

#connected component analysis then point extraction (endpoint or midpoint? (or both))
#hough transform 
