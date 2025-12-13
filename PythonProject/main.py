import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from datetime import datetime
from tensorflow.keras.models import load_model

from motion_detection import MotionDetector
from injury_logic import injury_decision



data = json.load(open(
    "/home/harribelsa/PycharmProjects/PythonProject/vehicle_data.json",
    encoding="utf-8"
))


model = load_model("model/injury_model.h5")
CLASSES = ["LOW", "MEDIUM", "HIGH"]

SPEED_MAP = {
    "LOW": 60,
    "MEDIUM": 130,
    "HIGH": 180,
    "UNKNOWN": 0
}

def classify_frame(frame):
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)[0]
    return CLASSES[np.argmax(pred)], float(np.max(pred))

def classify_image(path):
    img = cv2.imread(path)
    return classify_frame(img)

def classify_video(path):
    cap = cv2.VideoCapture(path)
    preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lvl, _ = classify_frame(frame)
        preds.append(CLASSES.index(lvl))
    cap.release()
    if not preds:
        return "UNKNOWN"
    return CLASSES[int(np.mean(preds))]

def calculate_crash_severity(speed_kmh):
    speed_ms = speed_kmh / 3.6
    stop_time = 0.15 if speed_kmh <= 60 else 0.10 if speed_kmh <= 130 else 0.07
    g_force = speed_ms / (stop_time * 9.81)

    if g_force < 20:
        severity = "LOW"
    elif g_force < 50:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return {
        "speed": speed_kmh,
        "g_force": round(g_force, 2),
        "stop_time": stop_time,
        "severity": severity
    }


def generate_report(level, reason, sensors, speed, crash):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
===== Accident Medical Report =====
Date & Time: {now}

--- AI & Crash Analysis ---
Injury Level: {level}
Reason: {reason}
Vehicle Speed: {speed} km/h
Estimated G-Force: {crash['g_force']} G
Impact Stop Time: {crash['stop_time']} sec
Crash Severity: {crash['severity']}

--- Sensors ---
Heart Rate: {sensors['heart']} bpm
Pressure: {sensors['pressure']} PSI
Temperature: {sensors['temp']} °C

--- Vehicle Owner ---
Name: {data['owner']['name']}
ID: {data['owner']['id']}
Phone: {data['owner']['phone']}

--- Vehicle Info ---
Type: {data['vehicle']['type']}
Plate: {data['vehicle']['plate']}
Registration: {data['vehicle']['registration']}
Insurance: {data['vehicle']['insurance']}

==================================
"""
    with open("accident_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    messagebox.showinfo("تم الحفظ", "تم إنشاء التقرير بنجاح")


class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emergency Crash Dashboard")
        self.root.geometry("1200x800")

        self.cap = cv2.VideoCapture(0)
        self.motion_detector = MotionDetector()

        self.analysis_count = 0
        self.last_level = "UNKNOWN"
        self.injury_reason = ""
        self.current_speed = 0
        self.crash_data = {}

        self.sensors = {
            "heart": {"value": 80},
            "pressure": {"value": 120},
            "temp": {"value": 36.5}
        }

        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
        video_frame = tk.Frame(self.root, bd=2, relief="ridge")
        video_frame.place(x=20, y=20, width=640, height=480)
        tk.Label(video_frame, text="Live Camera", font=("Arial", 14, "bold")).pack()
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        level_frame = tk.Frame(self.root, bd=2, relief="ridge")
        level_frame.place(x=680, y=20, width=480, height=120)
        tk.Label(level_frame, text="Risk Level", font=("Arial", 16, "bold")).pack()
        self.level_label = tk.Label(level_frame, text="UNKNOWN",
                                    font=("Arial", 26, "bold"))
        self.level_label.pack(pady=10)

        sensor_frame = tk.Frame(self.root, bd=2, relief="ridge")
        sensor_frame.place(x=680, y=160, width=480, height=220)
        tk.Label(sensor_frame, text="Sensors", font=("Arial", 16, "bold")).pack()
        self.sensor_text = tk.Label(sensor_frame, font=("Courier", 12), justify="left")
        self.sensor_text.pack(pady=10)

        control_frame = tk.Frame(self.root, bd=2, relief="ridge")
        control_frame.place(x=680, y=400, width=480, height=140)
        tk.Button(control_frame, text="Upload Image", command=self.load_image)\
            .grid(row=0, column=0, padx=20, pady=10)
        tk.Button(control_frame, text="Upload Video", command=self.load_video)\
            .grid(row=0, column=1, padx=20, pady=10)
        tk.Button(control_frame, text="Save Report", command=self.save_report)\
            .grid(row=1, column=0, columnspan=2, pady=15)

        stats_frame = tk.Frame(self.root, bd=2, relief="ridge")
        stats_frame.place(x=20, y=520, width=1140, height=240)
        tk.Label(stats_frame, text="Live Stats",
                 font=("Arial", 16, "bold")).pack()
        self.stats_text = tk.Text(stats_frame, height=10, width=130)
        self.stats_text.pack()

    def update_sensors(self):
        self.sensors["heart"]["value"] = np.random.randint(60, 150)
        self.sensors["pressure"]["value"] = np.random.randint(90, 180)
        self.sensors["temp"]["value"] = round(np.random.uniform(36, 40), 1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (600, 400))

            motion = self.motion_detector.detect(frame)

            # القرار الطبي النهائي
            self.last_level, self.injury_reason = injury_decision(frame, motion)

            self.current_speed = SPEED_MAP.get(self.last_level, 0)
            self.crash_data = calculate_crash_severity(self.current_speed)

            self.level_label.config(text=self.last_level)
            self.update_sensors()

            sensor_output = ""
            for k, v in self.sensors.items():
                sensor_output += f"{k.upper():10}: {v['value']}\n"
            self.sensor_text.config(text=sensor_output)

            cv2.putText(frame, f"Speed: {self.current_speed} km/h",
                        (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"G-Force: {self.crash_data['g_force']} G",
                        (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Injury: {self.injury_reason}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(
                tk.END,
                f"Analysis Count: {self.analysis_count}\n"
                f"Risk Level: {self.last_level}\n"
                f"Injury Reason: {self.injury_reason}\n"
                f"Speed: {self.current_speed} km/h\n"
                f"G-Force: {self.crash_data['g_force']} G\n"
            )

        self.root.after(40, self.update_frame)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            lvl, _ = classify_image(path)
            self.analysis_count += 1
            messagebox.showinfo("Result", f"Level: {lvl}")

    def load_video(self):
        path = filedialog.askopenfilename()
        if path:
            lvl = classify_video(path)
            self.analysis_count += 1
            messagebox.showinfo("Result", f"Level: {lvl}")

    def save_report(self):
        generate_report(
            self.last_level,
            self.injury_reason,
            {
                "heart": self.sensors["heart"]["value"],
                "pressure": self.sensors["pressure"]["value"],
                "temp": self.sensors["temp"]["value"]
            },
            self.current_speed,
            self.crash_data
        )


root = tk.Tk()
app = DashboardApp(root)
root.mainloop()
