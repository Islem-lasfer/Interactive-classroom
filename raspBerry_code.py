import cv2
import face_recognition
from fpdf import FPDF
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import pocketsphinx
import time
import mediapipe as mp
import numpy as np
import threading
from ultralytics import YOLO
import pytesseract
from transformers import pipeline
import matplotlib.pyplot as plt
import tempfile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from speech_recognition import Recognizer, Microphone, WaitTimeoutError
import speech_recognition as sr
from datetime import datetime

# Initialize YOLO model
best_model = YOLO('best.pt')

# Initialize video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Unable to open camera.")

running = True
spoken_text = ""
spoken_text_lock = threading.Lock()
recognizer = sr.Recognizer()

def binaryImage(img):
    if isinstance(img, str):
        image = cv2.imread(img)
        if image is None:
            raise ValueError("The image could not be loaded. Check the file path.")
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise TypeError("Input must be an image path (str) or a loaded image (numpy.ndarray).")

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_black_pixels = np.sum(binary_image == 0)
    num_white_pixels = np.sum(binary_image == 255)

    if num_black_pixels > num_white_pixels:
        print("Inversion of colors because black is the majority...")
        binary_image = cv2.bitwise_not(binary_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Image binaire")
    plt.imshow(binary_image, cmap="gray")
    plt.axis("off")
    plt.show()

    return binary_image

def process_and_binarize_image(image):
    if image is None:
        print("Error: Unable to process image.")
        return None

    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = 0
    color_only_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    grayscale_image = cv2.cvtColor(color_only_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_black_pixels = np.sum(binary_image == 0)
    num_white_pixels = np.sum(binary_image == 255)

    if num_black_pixels > num_white_pixels:
        print("Inversion of colors because black is the majority...")
        binary_image = cv2.bitwise_not(binary_image)
    """
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Image sans luminance")
    plt.imshow(cv2.cvtColor(color_only_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Image binaire (inversée si nécessaire)")
    plt.imshow(binary_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    """
    return binary_image

def capture_audio():
    global spoken_text, recognizer
    while running:
        with sr.Microphone() as source:
            try:
                print("Adjusting ambient noise...")
                recognizer.adjust_for_ambient_noise(source)
                print("Listen...")
                audio = recognizer.listen(source, timeout=7)
                print("Acknowledgement...")
                text = recognizer.recognize_google(audio, language="en-US")
                spoken_text += text + ". "
                print(f"Recognized text: {spoken_text}")
            except sr.WaitTimeoutError:
                print("Listening timeout exceeded.")
            except Exception as e:
                print(f"Speech recognition error: {e}")

threading.Thread(target=capture_audio, daemon=True).start()

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "course", align="C", ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def nettoyer_texte(texte):
    remplacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "é": "e", "è": "e", "à": "a", "ç": "c", "ù": "u"
    }
    for ancien, nouveau in remplacements.items():
        texte = texte.replace(ancien, nouveau)
    return texte

def add_content(pdf, text, image_path):
    text = nettoyer_texte(text)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)

    if image_path is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_image_path = temp_file.name
            cv2.imwrite(temp_image_path, image_path)

        try:
            pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=100)
            pdf.ln(60)
        except RuntimeError:
            pdf.cell(0, 10, "Error: Unable to load processed image.", ln=True)

        pdf.ln()
        pdf.ln()

pdf2 = PDF()
pdf2.add_page()

def is_included(inner, outer):
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

def YOLO_detection(frame):
    global spoken_text

    spoken_text += "\n"
    local_text = spoken_text
    spoken_text = ""

    add_content(pdf2, local_text, None)

    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    corrector = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

    sr_image_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sr_image = sr_image_bgr

    binary_image = process_and_binarize_image(frame)
    results = best_model.predict(source=sr_image_bgr, conf=0.1)

    boxes_with_classes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        boxes_with_classes.append((x1, y1, x2, y2, cls))

    filtered_boxes_with_classes = []
    for i, box1 in enumerate(boxes_with_classes):
        included = False
        for j, box2 in enumerate(boxes_with_classes):
            if i != j and is_included(box1[:4], box2[:4]):
                included = True
                break
        if not included:
            filtered_boxes_with_classes.append(box1)

    sorted_boxes_with_classes = sorted(filtered_boxes_with_classes, key=lambda box: box[1])

    text_images_dict = {}
    for i, (x1, y1, x2, y2, cls) in enumerate(sorted_boxes_with_classes):
        cropped_image = sr_image[y1:y2, x1:x2]
        if cls == 3:
            cropped_image = binaryImage(cropped_image)
            custom_config = r'--oem 1 --psm 6 --tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata"'
            detected_text = pytesseract.image_to_string(cropped_image, lang='eng', config=custom_config).strip()
            corrected_text = corrector(f"grammar: {detected_text}")[0]['generated_text']
            text_images_dict[f'image_{i}'] = {
                'image': cropped_image,
                'detected_text': detected_text,
                'corrected_text': corrected_text
            }
            if corrected_text and (len(corrected_text) + 5) < len(detected_text) and corrected_text != "" and corrected_text != "False" and corrected_text != "True":
                add_content(pdf2, corrected_text, None)
            else:
                add_content(pdf2, detected_text, None)
        else:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=7, C=2)
            processed_image = cv2.medianBlur(processed_image, 5)
            add_content(pdf2, "", processed_image)

known_face_encodings = []
known_face_names = []
detected_faces = {}
displayed_faces = set()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def load_known_faces():
    image = face_recognition.load_image_file("CCI04102024_0002.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("islem Lasfer")

def detect_persons(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(image)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        names.append(name)

    return names, face_locations, face_landmarks_list
def generate_pdf(names, pdf_path):
    attendance_status = {name: "Absent" for name in known_face_names}
    for name in names:
        if name in attendance_status:
            attendance_status[name] = "Present"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_xy(0, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(210, 10, "Attendance list", 0, 1, 'C')
    current_date = datetime.now().strftime("%m/%d/%Y")
    pdf.set_xy(0, 20)
    pdf.set_font("Arial", size=12)
    pdf.cell(210, 10, current_date, 0, 1, 'C')
    pdf.set_xy(10, 40)
    pdf.set_font("Arial", size=12)

    # Create table header
    pdf.cell(100, 10, "Name", 1)
    pdf.cell(100, 10, "Status", 1)
    pdf.ln()

    # Add names and their attendance status to the table
    for name, status in attendance_status.items():
        pdf.cell(100, 10, name, 1)
        pdf.cell(100, 10, status, 1)
        pdf.ln()

    pdf.output(pdf_path)

def on_download_button_click():
    threading.Thread(target=download_pdf).start()

def download_pdf():
    names = [table.item(item, "values")[0] for item in table.get_children()]
    if "Unknown" in names:
        names.remove("Unknown")
    pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if pdf_path:
        threading.Thread(target=generate_pdf, args=(names, pdf_path), daemon=True).start()
        messagebox.showinfo("Succès", "The PDF was generated successfully.")

alpha = 1.0
fade_out = True

def show_frame():
    global alpha, fade_out
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        names, face_locations, face_landmarks_list = detect_persons(frame_rgb)

        current_time = time.time()
        for (top, right, bottom, left), name in zip(face_locations, names):
            detected_faces[(top, right, bottom, left)] = current_time
            if name not in displayed_faces:
                if name == "Unknown":
                    table.insert("", "end", values=(name, "Unknown person detected"), tags=("Unknown",))
                else:
                    table.insert("", "end", values=(name, "Known person detected"))
                displayed_faces.add(name)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        overlay = frame_rgb.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=1
                    )
                )

        frame_rgb = cv2.addWeighted(overlay, alpha, frame_rgb, 1 - alpha, 0)

        if fade_out:
            alpha -= 0.01
            if alpha <= 0.1:
                fade_out = False
        else:
            alpha += 0.01
            if alpha >= 1:
                fade_out = True

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    camera_label.after(100, show_frame)

threading.Thread(target=load_known_faces, daemon=True).start()

def capture_image2():
    def task():
        cap2 = cv2.VideoCapture(0)
        ret, frame = cap2.read()
        if not ret:
            print("Error: Unable to capture image.")
        cap2.release()        
        #frame = cv2.imread("WIN_20250208_21_15_14_Pro.jpg")
        threading.Thread(target=YOLO_detection, args=(frame,), daemon=True).start()
    threading.Thread(target=task, daemon=True).start()

def telecharger_cours():
    def task():
        pdf_path2 = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if pdf_path2:
            pdf2.output(pdf_path2)
            messagebox.showinfo("Succès", "The PDF has been generated successfully.")
    threading.Thread(target=task, daemon=True).start()

def on_close():
    def task():
        global running
        running = False
        root.destroy()
        root.quit()
    threading.Thread(target=task, daemon=True).start()

root = tk.Tk()
root.title("Interactive classroom")

menu_frame = tk.Frame(root, bg="orange", width=200)
menu_frame.pack(side="left", fill="y")

download_button = tk.Button(menu_frame, text="Download the attendance PDF", command=on_download_button_click)
download_button.pack(pady=10, padx=10, fill="x")

capture_button = tk.Button(menu_frame, text="Capture the board", command=capture_image2)
capture_button.pack(pady=10, padx=10, fill="x")

cours_button = tk.Button(menu_frame, text="Download the course", command=telecharger_cours)
cours_button.pack(pady=10, padx=10, fill="x")

quit_button = tk.Button(menu_frame, text="To leave", command=on_close)
quit_button.pack(pady=10, padx=10, fill="x")

camera_frame = tk.Frame(root)
camera_frame.pack(side="top", fill="both", expand=True)

camera_label = tk.Label(camera_frame)
camera_label.pack()

table_frame = tk.Frame(root)
table_frame.pack(side="bottom", fill="both", expand=True)

table = ttk.Treeview(table_frame, columns=("name", "status"), show="headings")
table.heading("name", text="name")
table.heading("status", text="status")
table.pack(fill="both", expand=True)

table.tag_configure("Unknown", foreground="red")

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

threading.Thread(target=show_frame, daemon=True).start()

root.mainloop()

cap.release()
cv2.destroyAllWindows()