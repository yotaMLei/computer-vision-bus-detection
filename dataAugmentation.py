"""
Computer Vision course Project ,January 2020
Yotam Leibovitz 204095632
Eyal Asulin     300037397
dataAugmentation file creates a large data set for the model training,
by performing augmentations on the original image set
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Classes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
class IMAGE:

    def __init__(self):
        self.ROIS = []

    def set_image(self, im):
        self.I = imageio.imread(im)
        self.imShape = self.I.shape

    def clear_ROIS(self):
        self.ROIS = []

    def add_ROI(self, pos):
        self.ROIS.append(pos)

    def show_ROI(self, title, edgecolor, numGT, saveDir=None):
        fig, ax = plt.subplots(1)
        ax.imshow(self.I)
        if (not isinstance(edgecolor, list) and len(self.ROIS) > 0):
            edgecolor = [edgecolor] * len(self.ROIS)
        for i in range(0, numGT):
            ROI = self.ROIS[i]
            rect = patches.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth=1, edgecolor=edgecolor[i],
                                     facecolor='none')
            ax.add_patch(rect)
        for i in range(numGT, len(self.ROIS)):
            ROI = self.ROIS[i]
            rect = patches.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth=1, edgecolor=edgecolor[i],
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
        # if(saveDir is None):
        #     ax.text(15,160,text, fontdict = fontdict, bbox={'facecolor':'yellow', 'edgecolor':'yellow','alpha':0.5, 'pad':2})
        # else:
        #     ax.text(15, 300, text, fontdict=fontdict,bbox={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.5, 'pad': 2})
        plt.title(title)
        if (not saveDir is None):
            plt.savefig(os.path.join(saveDir, title), dpi=500)
        plt.close()

    def close(self):
        plt.close()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def create_annotations_file_project(filename, images_names, bbs_list, bbs_color):
    ann_file = open(filename, 'a')

    for i in range(len(images_names)):
        if images_names[i][:-4] == '.JPG':
            ann_file.write(images_names[i] + ':')
        else:
            ann_file.write(images_names[i][:-4] + '.JPG:')

        for j in range(len(bbs_list[i].bounding_boxes)):
            annsGT_aug = [int(bbs_list[i].bounding_boxes[j].x1),
                          int(bbs_list[i].bounding_boxes[j].y1),
                          int(bbs_list[i].bounding_boxes[j].width),
                          int(bbs_list[i].bounding_boxes[j].height),
                          bbs_color[j]]
            ann_file.write(str(annsGT_aug))
            if j < len(bbs_list[i].bounding_boxes) - 1:
                ann_file.write(',')
        ann_file.write('\n')
    ann_file.close()


def create_annotations_file_ssd(filename, images_names, bbs_list, bbs_color):
    ann_file = open(filename, 'a')
    for i in range(len(images_names)):

        ## According to SSD input format
        for j in range(len(bbs_list[i].bounding_boxes)):
            str1 = (images_names[i] + "," + str(bbs_list[i].bounding_boxes[j].x1_int) + \
                    "," + str(bbs_list[i].bounding_boxes[j].x2_int) + \
                    "," + str(bbs_list[i].bounding_boxes[j].y1_int) + \
                    "," + str(bbs_list[i].bounding_boxes[j].y2_int) + "," + str(bbs_color[j]) + "\n")
            ann_file.writelines(str1)
    ann_file.close()

    # bbs_aug[0].bounding_boxes[j].y1_int
    # bbs = BoundingBoxesOnImage(bbs_aug[i].bounding_boxes, shape=image.shape)
    # ia.imshow(bbs.draw_on_image(images_aug[i], size=2, color=MAGENTA))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Locations & Definitions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
numOfAugmentationPerImg = 20
SHOW_IMAGES = False # show the augmented images for testing
RATIO = 512 / 3648  # translate bounding boxes of original images to 512X512 images
DATA_SET_NUM = 8

annFileNameGT = 'annotationsTrain.txt'
annFileName_aug = 'annotationsTrain_aug.txt'
annFileNAmeGT_resize = 'annotationTrain_resize.txt'
# myAnnFileName = 'newAnns.txt'
busDir = r'busesTrain\resized'
saveDir = r'DATA sets\DATA set ' + str(DATA_SET_NUM)
annFileName_ssd = "ssd_annotations.csv"
objectsColors = {'g': '1', 'y': '2', 'w': '3', 's': '4', 'b': '5', 'r': '6'}
objectsColorsInv = {v: k for k, v in objectsColors.items()}
objectsColorsForShow = {'g': 'g', 'y': 'y', 'w': 'w', 's': 'tab:gray', 'b': 'b', 'r': 'r'}
image = IMAGE()
writtenAnnsLines = {}
# annFileEstimations = open(myAnnFileName, 'r')
annFileGT = open(annFileNameGT, 'r')
writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())
# writtenAnnsLines['Augmentation'] = (annFileEstimations.readlines())
images_orig = []
bbs_orig = []
images_names = []
bbs_orig_color = []

# if os.path.isfile(annFileName_ssd) :
#     os.remove(annFileName_ssd)
# file1 = open(annFileName_ssd,"a")

# create directories if needed
path = os.getcwd()
if not os.path.isdir(path + '\\' + busDir):
    os.mkdir(path + '\\' + busDir)
if not os.path.isdir(path + '\\' + saveDir):
    os.mkdir(path + '\\' + saveDir)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Augment training image set
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
for i in range(len(writtenAnnsLines['Ground_Truth'])):
    print('processing image number ' + str(i+1) + ' out of ' + str(len(writtenAnnsLines['Ground_Truth'])))
    # Get image name and ground truth annotations of the image
    lineGT = writtenAnnsLines['Ground_Truth'][i].replace(' ', '')
    colors = []
    imName = lineGT.split(':')[0]
    imName = imName[:-4]
    # for storing the bounding boxes
    bbs_list = []
    bbs_list_color = []

    bus = os.path.join(busDir, imName + '_resized3.jpg')
    image.set_image(bus)
    image.clear_ROIS()
    annsGT = lineGT[lineGT.index(':') + 1:].replace('\n', '')
    annsGT = ast.literal_eval(annsGT)
    if (not isinstance(annsGT, tuple)):
        annsGT = [annsGT]

    # evaluate bounding boxes from GT annotations
    for ann in annsGT:
        xmin, ymin, width, height = ann[0], ann[1], ann[2], ann[3]
        bb = BoundingBox(x1=xmin * RATIO, y1=ymin * RATIO, x2=(xmin + width) * RATIO, y2=(ymin + height) * RATIO)
        bbs_list.append(bb)
        bbs_list_color.append(ann[4])

    numGT = len(annsGT)
    # update image list
    images_orig.append(image.I)
    # update bounding boxes list
    bbs = BoundingBoxesOnImage(bbs_list, shape=image.I.shape)
    bbs_orig.append(bbs)
    # update image names list
    images_names.append(imName)
    # update ann color list
    bbs_orig_color.append(bbs_list_color)

    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # # perform augmentations on the image
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    sometimes = lambda aug: iaa.Sometimes(0.4, aug)

    seq = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=0.85),
        # iaa.AdditiveGaussianNoise(scale=(5, 10)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-30, 30)),
        iaa.Affine(scale=(0.5, 1.0)),
        iaa.GammaContrast((0.5, 2.5)),
        # iaa.LinearContrast((0.75, 1.5)),
        # sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.15)),
        # In some images distort local areas with varying strength.
        # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
        # sometimes(iaa.AddToHue(-10, 10)),
        # sometimes(iaa.AddToSaturation(-50, 50)),
        iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(1, iaa.Add((-50, 50)))),
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # iaa.Crop(percent=(0, 0.1)),
        # sometimes(iaa.CoarseDropout((0.05, 0.1), size_percent=(0.05, 0.1))),
        # sometimes(iaa.SaltAndPepper(0.025)),
        sometimes(iaa.GaussianBlur(sigma=(0.5, 1.0))),
    ], random_order=True)


    tempTuple = [seq(image=image.I, bounding_boxes=bbs) for _ in range(numOfAugmentationPerImg)]
    images_aug = [currTemp[0] for currTemp in tempTuple]
    bbs_aug = [currTemp[1] for currTemp in tempTuple]


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # save the augmented images
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    images_aug_names = []
    # test image
    if SHOW_IMAGES: ia.imshow(bbs.draw_on_image(image.I, size=4))

    for i in range(len(images_aug)):
        # test image
        if SHOW_IMAGES: ia.imshow(bbs_aug[i].draw_on_image(images_aug[i], size=4))

        augmented_file_name = imName + '_aug_' + str(i) + '.jpg'
        images_aug_names.append(augmented_file_name)
        imageio.imwrite(os.path.join(saveDir, augmented_file_name), images_aug[i])
        # save the original image in the Data set directory
        imageio.imwrite(os.path.join(saveDir, imName + '_resized3.jpg'), image.I)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # create new annotations files
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    create_annotations_file_project(annFileName_aug, images_aug_names, bbs_aug, bbs_list_color)
    create_annotations_file_ssd(annFileName_ssd, images_aug_names, bbs_aug, bbs_list_color)

    # add the annotations of the original image to ssd annotations file
    create_annotations_file_project(annFileNAmeGT_resize, [imName + '_resized3.jpg'], [bbs], bbs_list_color)
    create_annotations_file_ssd(annFileName_ssd, [imName + '_resized3.jpg'], [bbs], bbs_list_color)

# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# # close files
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
annFileGT.close()

