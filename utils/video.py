"""
Utility functions for videos, requires opencv
"""
from absl import app, flags, logging
import cv2
import os


def video_to_frames(video_path, frames_dir, stats_dir, overwrite=True):

    video_path, video_filename = os.path.split(video_path)

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    if os.path.exists(os.path.join(frames_dir, video_filename)) and not overwrite:
        logging.info("{} exists, won't overwrite.".format(os.path.join(frames_dir, video_filename)))
        return [os.path.join(frames_dir, video_filename, file)
                for file in os.listdir(os.path.join(frames_dir, video_filename))]
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    # load the video, ensure its valid
    capture = cv2.VideoCapture(os.path.join(video_path, video_filename))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:  # Might be a problem if video has no frames
        logging.error("Video has no frames. Check your opencv + ffmpeg installation, can't read videos!!!\n"
                      "You may need to install open cv by source not pip")

        return None

    # Let's go through the video and save the frames
    frames_list = list()
    for current in range(total):
        flag, frame = capture.read()
        if flag == 0:
            # print("frame %d error flag" % current)
            continue
        if frame is None:
            break
        height, width, _ = frame.shape
        cv2.imwrite(os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(current+1)), frame)
        frames_list.append(os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(current+1)))

    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, video_filename[:-4]+'.txt'), 'w') as f:
        f.write('{},{},{},{}'.format(video_filename, width, height, current))

    return frames_list


def frames_to_video():
    return NotImplementedError
