from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import time
from src.core import UniversalGhostPatch
from src.utils import preprocess_frame, tensor_to_frame

app = Flask(__name__)

# --- Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

try:
    attacker = UniversalGhostPatch(device=DEVICE)
except Exception as e:
    print(f"Error initializing attacker: {e}")
    exit(1)

# Camera Setup
# Adjust index 0/1 depending on environment
# On Pi, /dev/video0 usually maps to index 0
camera = cv2.VideoCapture(0) 
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global State
state = {
    "attack_active": False,
    "fps": 0,
    "mode": "Monitoring"
}

def generate_frames():
    prev_time = 0
    while True:
        success, frame = camera.read()
        if not success:
            # Attempt reconnect
            camera.open(0)
            time.sleep(1)
            continue

        # FPS Calculation
        now = time.time()
        fps = 1 / (now - prev_time) if prev_time > 0 and (now - prev_time) > 0 else 0
        state["fps"] = round(fps, 1)
        prev_time = now

        try:
            if state["attack_active"]:
                # 1. Preprocess
                input_tensor = preprocess_frame(frame, DEVICE)
                
                # 2. Universal Ghost Logic (Saliency + Patch)
                adv_tensor = attacker.apply_patch(input_tensor)
                
                # 3. Postprocess
                frame_processed = tensor_to_frame(adv_tensor)
                
                # Rescale back to display size
                frame = cv2.resize(frame_processed, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            print(f"Processing Error: {e}")

        # Encode for web
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle', methods=['POST'])
def toggle():
    state["attack_active"] = not state["attack_active"]
    state["mode"] = "ATTACKING (Saliency Guidance)" if state["attack_active"] else "Monitoring"
    print(f"Attack State: {state['attack_active']}")
    return jsonify(state)

@app.route('/api/stats')
def stats():
    return jsonify(state)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
