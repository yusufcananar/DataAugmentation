import cv2
from scipy.ndimage import generic_filter
import numpy as np

I = cv2.imread("dataset/im_(1).jpg")
I_filt = generic_filter(I, np.std, size=3)
cv2.imshow("im",I_filt)
cv2.waitKey()
cv2.destroyAllWindows()