import cv2
def extract_frames(video_path, output_path, num_frames=90):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_skip = total_frames // num_frames

    success,image = vidcap.read()
    count = 0
    frame_number = 0

    while success:
        if frame_number % frames_to_skip == 0:
            cv2.imwrite(f"{output_path}/frame{count:04d}.jpg", image) 
            count += 1
        success,image = vidcap.read()
        frame_number += 1

        if count == num_frames:
            break

    vidcap.release()

video_file = 'D:/image_experience/LoFTR-master/demo/video2.mp4'
output_folder = 'freiburg_sequence'
extract_frames(video_file, output_folder, num_frames=90)