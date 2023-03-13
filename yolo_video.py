import sys
import argparse
import os

#####
import cv2
from yolo_f import YOLO as YOLOf

from timeit import default_timer as timer
from yolo import YOLO
from PIL import Image  

##test_dir = "dataset/road_data/train/arterial" # Here put your directory of images which needs to be detected
test_dir = "data/ori" # Here put your directory of images which needs to be detected
save_dir = "data/box/" # Here put your directory where you want to save the images

testfiles= os.listdir(test_dir)

def detect_img(yolo):
    start = timer()
    for i in range(len(testfiles)):
        img = os.path.join(test_dir,testfiles[i])
        
        ######
        txtName = os.path.join(save_dir, testfiles[i])
        txtName = txtName.replace(".png", ".txt")
        
        print ('file: ', img)
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            ######
            r_image = yolo.detect_image(image, txtName)
            
            
    end = timer()
    time_taken = end - start
    print("time taken : ", time_taken)
    yolo.close_session()
    
def detect_imgf(yolof):
    print("f isn't modified")
    
    mask = cv2.imread("data/f_mod/mask.jpg")
    img = cv2.imread("data/f_mod/sample.png")
    h, w, c = img.shape
    mask_re = cv2.resize(mask,dsize=(w,h))
    
    sample = cv2.bitwise_and(mask_re, img)
    
    
    yolof.detect_image(Image.fromarray(sample))

    yolof.close_session()



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video/',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    
    #####
    f_test = open("data/f_mod/f_mod.txt", 'r')
    check = int(f_test.readline())
    f_test.close
    if check == 0:
        detect_imgf(YOLOf(**vars(FLAGS)))
        print("new f is modified")
        
    else :
        if FLAGS.image:
            """
            Image detection mode, disregard any remaining command line arguments
            """
            print("Image detection mode")
            if "input" in FLAGS:
                print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
            detect_img(YOLO(**vars(FLAGS)))

