from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np

app = Flask(__name__,  static_folder='static')

# Load the model
with open("t_model.p", "rb") as f:
    model = pickle.load(f)

# Function to preprocess the frames
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = cv2.equalizeHist(frame)                 # Equalize histogram
    frame = frame / 255.0                           # Normalize
    frame = cv2.resize(frame, (32, 32))             # Resize to model input size
    frame = np.expand_dims(frame, axis=(0, -1))     # Add batch and channel dimensions
    return frame

# Function to classify a frame
def classify_frame(frame):
    prediction = model.predict(frame)
    class_index = np.argmax(prediction)
    probability = np.max(prediction)
    return class_index, probability

# Function to map class index to class name
def get_class_name(class_index):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons', 
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 
        'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
        'No entry', 'General caution', 'Dangerous curve to the left', 
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 
        'Slippery road', 'Road narrows on the right', 'Road work', 
        'Traffic signals', 'Pedestrians', 'Children crossing', 
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
        'End of all speed and passing limits', 'Turn right ahead', 
        'Turn left ahead', 'Ahead only', 'Go straight or right', 
        'Go straight or left', 'Keep right', 'Keep left', 
        'Roundabout mandatory', 'End of no passing', 
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[class_index]

# Video stream generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame and get prediction
        processed_frame = preprocess_frame(frame)
        class_index, probability = classify_frame(processed_frame)
        class_name = get_class_name(class_index)

        # Overlay classification info
        cv2.putText(frame, f"Class: {class_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Probability: {probability:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/homes')
def home():
    return render_template('home.html')

@app.route('/setting')
def settings():
    return render_template('setting.html')

@app.route('/log')
def logs():
    return render_template('log.html')

@app.route('/camera_feeds')
def camera_feed():
    return render_template('camera_feed.html')

@app.route('/account')
def accounts():
    return render_template('accounts.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
