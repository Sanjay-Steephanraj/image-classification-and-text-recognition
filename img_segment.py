import cv2
import numpy as np

class PerspectiveTransform:
    def __init__(self, img):
        self.img = img
        if self.img is None:
            raise ValueError("Error: Could not load image. Check the file path")
        
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def preprocess(self):
        #Applying Adaptive Thresholding and Morphological Transformation
        blur_img = cv2.GaussianBlur(self.gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #Applying Morphological operations
        kernal = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernal, iterations=2)

        self.binary = binary
        return binary
    
    def find_receipt_contour(self):
        #Detect receipt contour and return the largest four quadrilateral
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        #Find the largest quadrilateral
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                return approx
        return None
    
    def order_points(self, pts):
        #Order Points as: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] #Top-left
        rect[2] = pts[np.argmax(s)] #Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] #Top-right
        rect[3] = pts[np.argmax(diff)] #Bottom-left

        return rect
    
    def apply_perspective_transform(self, pts):
        #Applying perspective transform to get top-down view
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        #Compute width and height of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        #Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1 ,0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        #Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))

        return warped
    
    def scan_receipt(self):
        #Main function to scan receipts
        self.preprocess()
        receipt_contour = self.find_receipt_contour()

        if receipt_contour is not None:
            transformed_img = self.apply_perspective_transform(receipt_contour.reshape(4, 2))
            return transformed_img
        else:
            raise ValueError("No valid receipts found!!!")