from ultralytics import YOLO
from tqdm import tqdm
import cv2, os


class YoloInterface:
    def __init__(self, cfg):
        self.ckp= cfg['ckp']
        self.traker= cfg['tracker']
        self.save_vid= cfg['save_vid']

    
    def __call__(self, vid_path, save_dir):

        model= YOLO(model= self.ckp)
        video = cv2.VideoCapture(vid_path)
        frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.save_vid:
            width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output = cv2.VideoWriter(os.path.join(save_dir, 'output2.mp4'), fourcc, fps, (width, height))

        if video.isOpened() == False:
            print('[!] error opening the video')

        print('[+] tracking video...\n')
        pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
        frame_cnt= 0
        try:
            with open(os.path.join(save_dir, 'results2.txt'), 'w') as f:
                while video.isOpened():
                    # reading video frame by frame
                    ret, frame = video.read()
                    frame_cnt+=1
                    # ret: if it detects a frame
                    if ret == True:
                        # run the detection from yolo
                        results = model.track(frame, persist=True, show= True, tracker= self.traker)
                        annotated_frame= results[0].plot()         
        
                        # cv2.imshow("Tracking", annotated_frame)
                        for r in results:
                            boxes= r.boxes

                            for box in boxes:
                                b = box.xyxy[0]
                                cls= box.cls
                                id= box.id
                                f.write(f"{b[0].item()} {b[1].item()} {b[2].item()} {b[3].item()} {cls.item()} {id.item()} {frame_cnt}\n")
                                    
                        if self.save_vid:
                            output.write(annotated_frame)

                        pbar.update(1)
                    else:
                        break
        except KeyboardInterrupt:
            pass

        pbar.close()
        video.release()
        if self.save_vid:
            output.release()

        print("Frame count: ", frame_cnt)
        print("original Frame count: ", frames_count)


if __name__ == "__main__":

    config= {'ckp': 'path/to/custom_yolo/weights',
             'tracker': 'botsort.yaml', 'save_vid': True}

    model= YoloInterface(config)
    model('paht/to/original/clip', 'output/path')

    