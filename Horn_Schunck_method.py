import numpy as np
import cv2

def horn_schunck(img1, img2, alpha=1.0, num_iterations=100):
    u = np.zeros_like(img1, dtype=np.float32)
    v = np.zeros_like(img1, dtype=np.float32)

    I_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)


    for _ in range(num_iterations):
        # Calculate temporal gradient
        I_t = img2 - img1 + I_x * u + I_y * v

        # Calculate flow update
        u_avg = cv2.boxFilter(u, -1, (5, 5))
        v_avg = cv2.boxFilter(v, -1, (5, 5))

        numerator = I_x * I_x * u_avg + I_x * I_y * v_avg + I_x * I_t
        denominator = I_x * I_x + I_y * I_y + alpha

        u = u_avg - numerator / denominator

        numerator = I_y * I_x * u_avg + I_y * I_y * v_avg + I_y * I_t
        denominator = I_x * I_x + I_y * I_y + alpha

        v = v_avg - numerator / denominator

    return u, v

