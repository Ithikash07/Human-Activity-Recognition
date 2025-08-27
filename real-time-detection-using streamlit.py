import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import torch
import time

# ==== Configurations ====
FRAME_SIZE = (224, 224)
N_FRAMES = 16
FRAME_STEP = 2  # frames interval for buffer update
CONFIDENCE_THRESHOLD = 0.4

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
    # Load YOLOv8 model (person class only)
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo.classes = [0]  # person class only
    har_model = tf.keras.models.load_model("path_to_your_trained_model.h5")
    return yolo, har_model

def format_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = frame.astype(np.float32) / 255.0
    return frame

def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), w)
    y2 = min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, FRAME_SIZE)
    return crop

def detect_person(yolo, frame):
    results = yolo(frame)
    if results is None or len(results.xyxy[0]) == 0:
        return None
    persons = [det.cpu().numpy() for det in results.xyxy[0] if det[4] >= CONFIDENCE_THRESHOLD]
    if len(persons) == 0:
        return None
    # Select the largest bounding box (most prominent person)
    areas = [(det[2]-det[0])*(det[3]-det[1]) for det in persons]
    idx = int(np.argmax(areas))
    bbox = persons[idx][:4]  # x1, y1, x2, y2
    return bbox

def main():
    st.title("Real-Time Human Activity Recognition (Single Person)")
    
    yolo, har_model = load_models()
    
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    prediction_text = st.empty()
    confidence_text = st.empty()

    frame_buffer = []
    bbox = None
    last_pred_time = 0
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam")
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update detection every N frames or if no bbox
        if len(frame_buffer) == 0 or len(frame_buffer) % FRAME_STEP == 0 or bbox is None:
            bbox = detect_person(yolo, rgb_frame)
        
        # Append cropped and formatted frame (or blank if no bbox)
        if bbox is not None:
            crop = crop_person(rgb_frame, bbox)
        else:
            crop = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        frame_buffer.append(crop)
        
        # Keep buffer size fixed
        if len(frame_buffer) > N_FRAMES:
            frame_buffer.pop(0)
        
        # Run prediction every 0.5 seconds if buffer full
        current_time = time.time()
        if len(frame_buffer) == N_FRAMES and (current_time - last_pred_time) > 0.5:
            input_clip = np.array(frame_buffer)
            input_clip = np.expand_dims(input_clip, axis=0)  # batch dimension
            preds = har_model.predict(input_clip)
            preds_prob = tf.nn.softmax(preds[0]).numpy()
            idx = int(np.argmax(preds_prob))
            conf = preds_prob[idx]
            action_label = UCF_CLASSES[idx]
            last_pred_time = current_time
            prediction_text.markdown(f"### Predicted Action: **{action_label}**")
            confidence_text.markdown(f"Confidence: {conf*100:.2f}%")
        
        # Draw bbox and label on original frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            if 'action_label' in locals():
                cv2.putText(frame, f'{action_label} ({conf*100:.1f}%)',
                            (x1, max(30, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()

if __name__ == "__main__":
    main()
