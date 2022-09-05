# This is a modification of the code from 
# https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import os
import itertools
import shutil
import math
import imutils


DEBUG = False


def getpoints_equally_distributed(img):
    points = []
    # Corners
    points.append([0, 0])
    points.append([0, img.shape[1]-1])
    points.append([img.shape[0]-1, 0])
    points.append([img.shape[0]-1, img.shape[1]-1])
    # Centres
    points.append([0, int((img.shape[1]-1)/2)])
    points.append([img.shape[0]-1, int((img.shape[1]-1)/2)])
    points.append([int((img.shape[0]-1)/2), 0])
    points.append([int((img.shape[0]-1)/2), img.shape[1]-1])
    # Quarters
    points.append([0, int((img.shape[1]-1)/4)])
    points.append([img.shape[0]-1, int((img.shape[1]-1)/4)])
    points.append([int((img.shape[0]-1)/4), 0])
    points.append([int((img.shape[0]-1)/4), img.shape[1]-1])
    points.append([0, 3*int((img.shape[1]-1)/4)])
    points.append([img.shape[0]-1, 3*int((img.shape[1]-1)/4)])
    points.append([3*int((img.shape[0]-1)/4), 0])
    points.append([3*int((img.shape[0]-1)/4), img.shape[1]-1])

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contour_points, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        n = len(contour_points[0])
        step = n/20
        
        # cv2.imshow('Image',gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('Image',thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        s = 0
        while s < 20:
            i = math.floor(s*step)
            p0 = contour_points[0][i][0][0]
            p1 = contour_points[0][i][0][1]
            points.append([p1, p0])
            s = s + 1

        # print(len(points),'points')

    except Exception as e:
        print(e)

    return points


""" def getpoints(img, fc, cc):
    points = []
    # Corners
    points.append([0, 0])
    points.append([0, img.shape[1]-1])
    points.append([img.shape[0]-1, 0])
    points.append([img.shape[0]-1, img.shape[1]-1])
    # Centres
    points.append([0, int((img.shape[1]-1)/2)])
    points.append([img.shape[0]-1, int((img.shape[1]-1)/2)])
    points.append([int((img.shape[0]-1)/2), 0])
    points.append([int((img.shape[0]-1)/2), img.shape[1]-1])
    # Quarters
    points.append([0, int((img.shape[1]-1)/4)])
    points.append([img.shape[0]-1, int((img.shape[1]-1)/4)])
    points.append([int((img.shape[0]-1)/4), 0])
    points.append([int((img.shape[0]-1)/4), img.shape[1]-1])
    points.append([0, 3*int((img.shape[1]-1)/4)])
    points.append([img.shape[0]-1, 3*int((img.shape[1]-1)/4)])
    points.append([3*int((img.shape[0]-1)/4), 0])
    points.append([3*int((img.shape[0]-1)/4), img.shape[1]-1])

    # Horizontal extremes
    n = 0
    while img[fc,n,0] <= 5:
        n += 1
    points.append([fc,n])
    n = img.shape[1]-1
    while img[fc,n,0] <= 5:
        n -= 1
    points.append([fc,n])
    # Vertical extremes
    n = 0
    while img[n,cc,0] <= 5:
        n += 1
    points.append([n,cc])
    n = img.shape[0]-1
    while img[n,cc,0] <= 5:
        n -= 1
    points.append([n,cc])

    try:
        k=1
        while k<=4:#[1:4]
            # right-down
            i = fc
            j = cc
            while i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1] and img[i,j,0] >= 250:
                i += 1
                j += k
            points.append([i-1,j-k])
            # right-up
            i = fc
            j = cc
            while i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1] and img[i,j,0] >= 250:
                i -= 1
                j += k
            points.append([i+1,j-k])
            # left-up
            i = fc
            j = cc
            while i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1] and img[i,j,0] >= 250:
                i -= 1
                j -= k
            points.append([i+1,j+k])
            # left-down
            i = fc
            j = cc
            while i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1] and img[i,j,0] >= 250:
                i += 1
                j -= k
            points.append([i-1,j+k])
            # Next angle
            k += 1
    except:
        print(str(4+4*(k-1)) + " points")

    #print(len(points))

    return points """


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)


# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
    # print(rect)


    # Insert points into subdiv
    for p in points:
        # print(p)
        # p = list(p)
        # if p[0]>=rect[2]:
        #     p[0]=rect[2]-1
        # if p[1]>=rect[3]:
        #     p[1]=rect[3]-1
        subdiv.insert((p[0], p[1]))




    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()
    
    # Find the indices of triangles in the points array

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            #x,y,z of triangle points
            for j in range(0, 3):
                #for each points
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    #print("triangles: ", delaunayTri)
    
    return delaunayTri


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv):

    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]),
                   3, (0, 0, 0), -1, cv2.LINE_AA, 0)


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img

#t1: punto della maschera img1
#t2: punto della maschera img2
#t3: punto medio
def morphTriangle(img1, img2, img, t1, t2, t, alpha):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    
    #plt.imshow(mask)
    #plt.show()
    
    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1] +
                                              r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask


def morphDiatoms(pathImg1Fourier, pathImg2Fourier, pathImg1, pathImg2, alpha):

    
    im = cv2.imread(pathImg1Fourier)
    pointsI1 = getpoints_equally_distributed(im)
    #for i in range(0, len(pointsI1)):
    #    plt.plot(pointsI1[i][1], pointsI1[i][0], marker='o', color='r')
    #plt.imshow(im)
    #plt.show()
    
    temp1 = im
    
    im2 = cv2.imread(pathImg2Fourier)
    im2 = cv2.resize(im2, (im.shape[1], im.shape[0]))
    pointsI2 = getpoints_equally_distributed(im2)
    #for i in range(0, len(pointsI2)):
    #    plt.plot(pointsI2[i][1], pointsI2[i][0], marker='o', color='r')
    #plt.imshow(im2)
    #plt.show()
    
    temp2 = im2

    im = cv2.imread(pathImg1)   #RGB image 1
    tam1 = im.shape             #dimension image 1
    im2 = cv2.imread(pathImg2)
    tam2 = im2.shape
    im2 = cv2.resize(im2, (im.shape[1], im.shape[0]))

    img1 = np.float32(im)
    img2 = np.float32(im2)
    
    points1 = []
    for p in pointsI1:
       points1.append((p[1], p[0]))
    points2 = []
    for p in pointsI2:
        points2.append((p[1], p[0]))

    points = []
    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((int(x), int(y)))
    # print(points)
    
    #check for points
    #for i in range(0, len(points)):
    #    plt.plot(points[i][0], points[i][1], marker='o', color='r')
    #plt.imshow(temp1)
    #plt.show()
    
    #for i in range(0, len(points)):
    #    plt.plot(points[i][0], points[i][1], marker='o', color='r')
    #plt.imshow(temp2)
    #plt.show()
    
    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)
    # space for mask
    imgMorphMask = np.zeros(img1.shape, dtype=img1.dtype)

    rectr = (0, 0, img1.shape[1], img1.shape[0])

    trir = calculateDelaunayTriangles(rectr, points)
    #print(trir) 

    for k in range(0, len(trir)):
        x = int(trir[k][0])
        y = int(trir[k][1])
        z = int(trir[k][2])

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x],  points[y],  points[z]]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
        morphTriangle(temp1, temp2, imgMorphMask, t1, t2, t, alpha)

    # Display Result
    if tam1[1] > tam2[1]:
        w = tam1[1] - alpha * np.abs(tam1[1]-tam2[1])
    elif tam1[1] < tam2[1]:
        w = tam1[1] + alpha * np.abs(tam1[1]-tam2[1])
    else:
        w = tam1[1]
    if tam1[0] > tam2[0]:
        h = tam1[0] - alpha * np.abs(tam1[0]-tam2[0])
    elif tam1[0] < tam2[0]:
        h = tam1[0] + alpha * np.abs(tam1[0]-tam2[0])
    else:
        h = tam1[0]
    imgMorph = cv2.resize(imgMorph, (int(w), int(h)))
    #cv2.imshow("Morphed", np.uint8(imgMorph))
    # cv2.waitKey(0)

    return np.uint8(imgMorph), np.uint8(cv2.resize(imgMorphMask,(int(w), int(h))))


def start_morph(original, fourier, output, alpha_min, alpha_max, alpha_increment, factor_increment):
    
    print("\nIf u asked for more image than possible, the program will create the maximum possible images")
    print("for create more image change alpha parameters\n")
    
    orig_path = original
    fourier_path = fourier
    out_path = output
    
    parts = []
    for item in os.listdir(orig_path):
        if os.path.isdir(orig_path + item):
            parts.append(item)
    
    #classes = os.listdir(os.path.join(orig_path, parts[0]))

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    segmented = [x for x in os.listdir(fourier_path)]
    for i in range(len(segmented)):
        segmented[i] = segmented[i].replace("_imag_res_fill.png", ".png")

    for part in parts:
        print("set: ", part)
        part_path = os.path.join(orig_path, part)
        os.makedirs(part_path.replace(orig_path, out_path), exist_ok=True)
        for clase in os.listdir(part_path):
            print("subclass: ", clase)
            class_path = os.path.join(part_path, clase)
            os.makedirs(class_path.replace(orig_path, out_path), exist_ok=True)
            images = os.listdir(class_path)
            for img in images: # Copy all original images
                img_path = os.path.join(class_path, img)
                shutil.copy(img_path, img_path.replace(
                    orig_path, out_path))

            if images == []:
                continue
            
            candidates = []
            for img in images:
                if img in segmented:
                    img_path = os.path.join(class_path, img)
                    candidates.append(img_path)
            combis = list(itertools.combinations(candidates, 2))
            
            #
            image_can_be_created = int(np.floor((alpha_max - alpha_min) / alpha_increment) + 1) * len(combis)
            image_wanted = int(len(images) * factor_increment)
            percentage = (image_wanted / image_can_be_created) * 100
            if percentage > 100:
                percentage = 100
            print("image that can be created: ", image_can_be_created)
            print("image you want to create: ", image_wanted)
            print("percentage of image created: ", percentage)
            #
            created = 0
            for c in combis:
                pathImg1Fourier = os.path.join(fourier_path, os.path.basename(
                    c[0]).replace(".png", "_imag_res_fill.png"))
                pathImg2Fourier = os.path.join(fourier_path, os.path.basename(
                    c[1]).replace(".png", "_imag_res_fill.png"))
                pathImg1 = c[0]
                pathImg2 = c[1]
                shutil.copyfile(pathImg1Fourier, 'tmp/im1f.png')
                shutil.copyfile(pathImg2Fourier, 'tmp/im2f.png')
                shutil.copyfile(pathImg1, 'tmp/im1.png')
                shutil.copyfile(pathImg2, 'tmp/im2.png')
                pathImg1Fourier_reg = 'tmp/im1f.png'
                pathImg2Fourier_reg = 'tmp/im2f.png'
                pathImg1_reg = 'tmp/im1.png'
                pathImg2_reg = 'tmp/im2.png'
                im1 = cv2.imread(pathImg1)
                im2 = cv2.imread(pathImg2_reg)
                im1f = cv2.imread(pathImg1Fourier)
                im2f = cv2.imread(pathImg2Fourier_reg)
                if im1 is None or im2 is None or im1f is None or im2f is None:
                    print("Error loading images")
                    print(pathImg1_reg)
                    print(pathImg2_reg)
                    print(pathImg1Fourier_reg)
                    print(pathImg2Fourier_reg)
                    exit(-1)
                if im1.shape[1] < im2.shape[1]:
                    aux = pathImg1Fourier
                    pathImg1Fourier = pathImg2Fourier
                    pathImg2Fourier = aux
                    aux = pathImg1
                    pathImg1 = pathImg2
                    pathImg2 = aux
                    aux = pathImg1Fourier_reg
                    pathImg1Fourier_reg = pathImg2Fourier_reg
                    pathImg2Fourier_reg = aux
                    aux = pathImg1_reg
                    pathImg1_reg = pathImg2_reg
                    pathImg2_reg = aux
                im1 = cv2.imread(pathImg1_reg)
                im2 = cv2.imread(pathImg2_reg)
                max_shape = list(im1.shape)
                if im2.shape[0] > im1.shape[0]:
                    max_shape[0] = im2.shape[0]
                if im2.shape[1] > im1.shape[1]:
                    max_shape[1] = im2.shape[1]
                all_imgs = np.zeros((11 * max_shape[0], max_shape[1], max_shape[2]), dtype=im1.dtype)
                
                alpha = alpha_min
                while alpha <= alpha_max:
                    if random.randrange(100) <= percentage:
                        created += 1
                        fname = pathImg1.replace(orig_path, out_path)
                        fname = fname.replace(".png", pathImg2[-8:-4] + "_" +
                                                "{:02d}".format(int(np.round(10 * alpha))) + ".png")
                        try:
                            new_img,new_img_mask = morphDiatoms(
                                pathImg1Fourier_reg, 
                                pathImg2Fourier_reg, 
                                pathImg1_reg, 
                                pathImg2_reg, 
                                alpha
                            )
                        except Exception as e:
                            print(e)
                            print('Morphing failed with pair:', pathImg1[-8:-4], pathImg2[-8:-4])
                        finally:
                            if DEBUG:
                                all_imgs[int(np.round(10 * alpha)) * max_shape[0]:int(np.round(
                                    10 * alpha)) * max_shape[0] + new_img.shape[0], 0:new_img.shape[1]] = new_img
                                cv2.imshow("Morphed", all_imgs)
                                cv2.waitKey(0)
                            else:
                                cv2.imwrite(fname, new_img)
                                
                                if not os.path.exists(out_path + '/mask'):
                                    os.makedirs(out_path + '/mask')
                                fname = out_path + 'mask/' + fname.rsplit('\\',1)[1]
                                cv2.imwrite(fname.replace(".png", "_mask.png"), new_img_mask)
                    alpha = alpha + alpha_increment
            print("created: ", created, '\n')