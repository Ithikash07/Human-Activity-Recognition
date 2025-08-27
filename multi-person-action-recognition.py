import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from bytetrack import BYTETracker
import tensorflow as tf
import tempfile
import os
from collections import deque

# ==== Configurations ====
FRAME_SIZE = (224, 224)
N_FRAMES = 16                 # Number of frames per clip for HAR
HAR_MODEL_PATH = 'VideoClassificationModel.h5'  # Your trained HAR model path
YOLO_MODEL_NAME = 'yolov8n.pt' 
PERSON_CLASS_ID = 0

# ==== UCF101 Classes from your HAR model ====
UCF_CLASSES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
              'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
              'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
              'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
              'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
              'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
              'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HandstandPushups',
              'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding',
              'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack',
              'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing',
              'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello',
              'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano',
              'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse',
              'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
              'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing',
              'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling',
              'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
              'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog',
              'WallPushups', 'WritingOnBoard', 'YoYo']

@st.cache_resource
def load_models():
    # Load YOLOv8 person detector model
    yolo_model = YOLO(YOLO_MODEL_NAME)
    # Initialize ByteTrack tracker
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    # Load trained HAR TensorFlow model
    har_model = tf.keras.models.load_model(HAR_MODEL_PATH)
    return yolo_model, tracker, har_model

def preprocess_frames(frames):
    frames = tf.convert_to_tensor(frames, dtype=tf.float32) / 255.0
    return tf.expand_dims(frames, axis=0)

def extract_person_clips(frames, bboxes, track_buffer, track_id):
    cropped_frames = []
    for frame, boxes_per_frame in zip(frames, bboxes):
        box = boxes_per_frame.get(track_id, None)
        if box is not None:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, FRAME_SIZE)
        else:
            crop = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        cropped_frames.append(crop)
    while len(cropped_frames) < N_FRAMES:
        cropped_frames.append(np.zeros((*FRAME_SIZE,3), dtype=np.uint8))
    return np.array(cropped_frames)[-N_FRAMES:]

def main():
    st.title("Multi-Person Human Activity Recognition")

    uploaded_file = st.file_uploader("Upload a video (.mp4 or .avi)", type=["mp4", "avi"])
    if not uploaded_file:
        st.info("Upload a video to start action recognition")
        return

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    yolo_model, tracker, har_model = load_models()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read video")
        return

    frame_buffer = deque(maxlen=N_FRAMES)
    bbox_buffer = deque(maxlen=N_FRAMES)
    stframe = st.empty()

    while ret:
        orig_frame = frame.copy()
        frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        yolo_results = yolo_model(frame_rgb)[0]

        person_detections = []
        for det in yolo_results.boxes.data.cpu().numpy():
            class_id = int(det[5])
            conf = det[4]
            if class_id == PERSON_CLASS_ID and conf > 0.5:
                x1, y1, x2, y2 = map(int, det[0:4])
                person_detections.append([x1, y1, x2, y2, conf])

        dets_np = np.array(person_detections)
        online_targets = tracker.update(dets_np, [frame.shape[1], frame.shape[0]])

        current_boxes = {}
        for t in online_targets:
            tid = int(t.track_id)
            bbox = (int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2]), int(t.tlbr[3]))
            current_boxes[tid] = bbox
            cv2.rectangle(orig_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(orig_frame, f'ID {tid}', (bbox[0], max(20,bbox[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        bbox_buffer.append(current_boxes)

        if len(frame_buffer) == N_FRAMES:
            for tid in bbox_buffer[0].keys():
                clip = extract_person_clips(list(frame_buffer), list(bbox_buffer), N_FRAMES, tid)
                input_clip = preprocess_frames(clip)
                preds = har_model.predict(input_clip)
                pred_label = int(np.argmax(preds))
                pred_conf = tf.nn.softmax(preds)[0][pred_label].numpy()
                label_text = f'{UCF_CLASSES[pred_label]}: {pred_conf:.2%}'
                bbox = bbox_buffer[-1].get(tid)
                if bbox:
                    x1, y1, _, _ = bbox
                    cv2.putText(orig_frame, label_text, (x1, max(40,y1-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        stframe.image(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        ret, frame = cap.read()

    cap.release()
    os.remove(video_path)

if __name__ == "__main__":
    main()
