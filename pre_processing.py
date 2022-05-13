import cv2
import os
# Opens the Video file

path = os.getcwd()+"/dataset"

def extract_foreground(frame):
    fgbg2 = cv2.createBackgroundSubtractorMOG2()
    mask = fgbg2.apply(frame)
    return mask

def preprocess():
    for filename in files(path):
        cap = cv2.VideoCapture(filename)

        j = 0
        i = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        div_num = round(length / 14) - 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i % div_num == 0:
                j += 1
                cv2.imwrite(os.getcwd()+f"/{filename}_frames/frame_" + str(j) + '.jpg', frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            try:
                os.mkdir(os.getcwd() + f"/dataset/{file}_frames")
            except FileExistsError:
                print("Directory esistente")
            filename = os.path.join("dataset", file)
            yield filename


if __name__ == '__main__':
    preprocess()