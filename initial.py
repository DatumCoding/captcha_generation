import cv2
import numpy as np
import os

def expected_values_back(image):
    exp = []
    for channel in cv2.split(image):
        vals = np.unique(channel , return_counts = True)
        ints , freqs = vals[0] , vals[1]
        freq_probs = freqs / np.sum(freqs)
        exp_vals = np.multiply(ints , freq_probs)
        exp.append(int(np.sum(exp_vals)))
    print(exp)
    return exp

def shear(t , rows , cols):
    shear = np.random.choice(10) / 10
    direction = np.random.choice(["x" , "y"] , 1 , p = [0.5 , 0.5])
    if(direction == "x"):
        m = np.float32([[1 , shear , 0] ,
                        [0 , 1 , 0] ,
                        [0 , 0 , 1]])
    else:
        m = np.float32([[1 , 0 , 0] ,
                        [shear , 1 , 0] ,
                        [0 , 0 , 1]])
    return cv2.warpPerspective(t.astype("uint8") , m , (int(cols * 1.5) , int(rows * 1.5)))

def rotation(t , rows , cols):
    angle = max(0 , np.radians(np.random.randn()) * 20)
    m = np.float32([[np.cos(angle) , -1 * (np.sin(angle)) , 0] ,
                    [np.sin(angle) , np.cos(angle) , 0] ,
                   [0 , 0 , 1]])
    return cv2.warpPerspective(t.astype("uint8") , m , (int(cols * 1.5) , int(rows * 1.5)))

def lines(t , rows , cols):
    x1 , y1 = 0 , np.random.choice(cols , 1)
    x2 , y2 = rows , np.random.choice(cols , 1)
    return cv2.line(t.astype("uint8"), (x1 , y1), (x2 , y2), (255, 255, 255), thickness = 1)

n = np.random.choice(range(5 , 7))
chars = np.random.choice(26 , n)
dirs = os.listdir('.//data')
real = ""
img_list = []
for c in chars:
    req_dir = dirs[c]
    real += req_dir
    pics = os.listdir(".//data//" + req_dir)
    num = np.random.randint(len(pics))
    img = cv2.imread(".//data//" + req_dir + "//" + pics[num])
    ret , thresh = cv2.threshold(img , 120 , 255 , cv2.THRESH_BINARY)
    rows , cols = img.shape[0] , img.shape[1]
    ig = shear(thresh , rows , cols)
    ig = rotation(ig , rows , cols)
    img_list.append(ig)
print(real)
temp = np.hstack(img_list)
em = np.where(temp == 0 , 1 , 0)
image_name = "lena.jpg"
image = cv2.imread(image_name)
back_exp = expected_values_back(image)
mask = np.multiply(em , np.array(back_exp))
temp += mask.astype("uint8")
temp = lines(temp , temp.shape[1] , temp.shape[0])
print(np.unique(temp))
cv2.imshow("test" , temp.astype("uint8"))
