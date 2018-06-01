import argparse
from yolo import YOLO
from yolo import detect_video



parser = argparse.ArgumentParser(description='Detect objects in videos')
parser.add_argument('-i', '--input', type=str, required=True, dest='input_video', help='Input path of videos')
parser.add_argument('-o', '--output', type=str, default='output.mp4', dest='output_video', help='Output destination of detected videos')

args = parser.parse_args()


if __name__ == '__main__':
    detect_video(YOLO(), args.input_video, args.output_video)
