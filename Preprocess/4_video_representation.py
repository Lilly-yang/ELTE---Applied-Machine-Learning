import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from comman_tools import *


def count_files(path, file_sudex):
    number = 0
    file_names = []
    for root, dirname, filenames in os.walk(path):
        for filename in sorted(filenames):
            # os.path.splitext()是一个元组,类似于('188739', '.jpg')，索引1可以获得文件的扩展名
            if os.path.splitext(filename)[1] == file_sudex and '._' not in filename:
                number += 1
                file_names.append(os.path.join(root, filename))

    return number, file_names


def img_plot(imgs, plot_number=6, title=''):
    plt.figure()
    plt.title(title)
    for i in range(plot_number):
        plt.subplot(1, plot_number, i + 1)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        plt.imshow(imgs[i])

    plt.show()


def data_argumentation(simpson_tra):
    # initialize an data augmenter as an 'empty' image data generator
    data_generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
                                        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                        zoom_range=0.2, horizontal_flip=True)
    # img_plot(simpson_tra, title='original')

    data_aurgued = next(data_generator.flow(simpson_tra, shuffle=False))
    # img_plot(data_aurgued, title='aurgument')

    return data_aurgued


images_path = '/Volumes/Li_YANG/Datasets/RAVDESS/preprocess'
resum = True

data_set = []
for root, dirs, files in os.walk(images_path):
    # coutral where to start
    if root == '/Volumes/Li_YANG/Datasets/RAVDESS/preprocess/Actor_02/01-01-07-02-02-02-02':
        resum = False

    if not resum:
        if files:
            images_array = []
            for file in files:
                if '.jpg' in file:
                    print('---process: ', root)

                    # Subsample the frames to reduce complexity (6 frames/video is enough)
                    img_number, img_paths = count_files(root, '.jpg')
                    sample_index = np.linspace(1, img_number, 6, dtype=np.int)
                    sample_img_paths = [img_paths[i - 1] for i in sample_index]

                    for ind, img_path in enumerate(sample_img_paths):
                        image = cv2.imread(img_path)
                        try:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB colorspace
                        except:
                            print('wrong file: %s' % (img_path))
                            exit()
                        # Plot our image using subplots to specify a size and title
                        # fig = plt.figure(figsize=(8, 8))
                        # ax1 = fig.add_subplot(111)
                        # ax1.set_xticks([])
                        # ax1.set_yticks([])
                        # ax1.set_title('Original Image')
                        # ax1.imshow(image)
                        # plt.show()
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert the RGB  image to grayscale

                        # Extract the faces from the images
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                        faces = face_cascade.detectMultiScale(gray)  # , 4, 6)
                        for (x, y, w, h) in faces:  # Get the bounding box for each detected face
                            face = image[y:y + h, x:x + w]

                        # Resize the face images to 64x64
                        face_reshape = cv2.resize(face, (64, 64))

                        images_array.append(face_reshape)
                        # Display the image with the detections
                        # fig = plt.figure(figsize=(8, 8))
                        # ax1 = fig.add_subplot(111)
                        # ax1.set_xticks([])
                        # ax1.set_yticks([])
                        # ax1.set_title(str(ind))
                        # ax1.imshow(face_reshape)
                        # plt.show()

                    # Apply data augmentation, and scaling [0, 1]
                    images_array = np.stack(images_array)
                    data_array_aurgument = data_argumentation(images_array)
                    data_array_aurgument /= 255.

                    # save to a tensor with shape (F,H,W,3) = (6,64,64,3)
                    data_array_aurgument = tf.convert_to_tensor(data_array_aurgument)
                    save_npy(data_array_aurgument, root, 'face_sample')

                    break
