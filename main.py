
import cv2
import albumentations as A
from PIL import Image
import numpy as np


image = cv2.imread("dataset/im_(1).jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [[0, 0, 300, 500,'oliveTree'],
          [0, 0, 270, 230,'oliveTree']]


# Create a pipline with 4 different transformations.
transform = A.Compose([
    # A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1))

# Apply transformation
# transformed = transform(image=image)

# Passing annotation coordinates and categories with the image
transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

print(transformed_bboxes)

cv2.rectangle(transformed_image, (int(transformed_bboxes[0][0]), int(transformed_bboxes[0][1])), (int(transformed_bboxes[0][2]), int(transformed_bboxes[0][3])),-1)
cv2.imshow("im", transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()