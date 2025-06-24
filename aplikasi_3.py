import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
import time
import cv2
import os
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class AnswerSheetDetector:
    """Enhanced YOLO-based answer sheet detector with consistent results and no sequential errors"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.class_names = ['A', 'B', 'C', 'D']  # Sesuaikan dengan class model Anda
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.is_loaded = True
            logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.is_loaded = False

    def detect_answers(self, image_path, confidence_threshold=0.5):
        """
        Detect answers dengan preprocessing yang konsisten
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Simpan gambar original untuk backup
        original_image = image.copy()
        
        # Enhanced preprocessing dengan parameter tetap
        # Gunakan parameter yang sama setiap kali
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        # Resize dengan ukuran target yang tetap, bukan faktor perkalian
        height, width = enhanced_image.shape[:2]
        target_width = width * 2
        target_height = height * 2
        enhanced_image = cv2.resize(enhanced_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        logger.debug(f"Image loaded and processed: original={original_image.shape}, processed={enhanced_image.shape}")
        
        # Run YOLO detection
        results = self.model(enhanced_image, conf=confidence_threshold)
        
        # Extract detections dengan pembulatan untuk konsistensi
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Bulatkan koordinat untuk konsistensi
                    detection = {
                        'bbox': [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                        'confidence': round(float(confidence), 4),
                        'class': class_name,
                        'center_x': round(float((x1 + x2) / 2), 2),
                        'center_y': round(float((y1 + y2) / 2), 2),
                        'width': round(float(x2 - x1), 2),
                        'height': round(float(y2 - y1), 2)
                    }
                    detections.append(detection)
        
        logger.debug(f"Raw detections: {len(detections)} objects detected")
        return detections, enhanced_image
    
    def filter_overlapping_detections(self, detections, iou_threshold=0.3):
        """
        Filter deteksi yang overlapping untuk menghindari duplikasi
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence descending
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered_detections = []
        
        for detection in detections_sorted:
            is_overlapping = False
            for filtered_detection in filtered_detections:
                iou = self.calculate_iou(detection['bbox'], filtered_detection['bbox'])
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def create_spatial_grid(self, detections, num_questions=40, questions_per_column=10):
        """
        Membuat grid spatial yang tidak terpengaruh sequential error
        """
        if not detections:
            return {}, {}
        
        num_columns = num_questions // questions_per_column
        
        # Extract coordinates
        x_coords = [d['center_x'] for d in detections]
        y_coords = [d['center_y'] for d in detections]
        
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Create stable column boundaries using clustering or division
        if len(set(x_coords)) >= num_columns:
            try:
                kmeans_x = KMeans(n_clusters=num_columns, random_state=42, n_init=10)
                x_clusters = kmeans_x.fit_predict(np.array(x_coords).reshape(-1, 1))
                column_centers = sorted(kmeans_x.cluster_centers_.flatten())
                
                # Create column boundaries
                column_boundaries = []
                for i in range(len(column_centers)):
                    if i == 0:
                        left_bound = x_min
                    else:
                        left_bound = (column_centers[i-1] + column_centers[i]) / 2
                    
                    if i == len(column_centers) - 1:
                        right_bound = x_max
                    else:
                        right_bound = (column_centers[i] + column_centers[i+1]) / 2
                    
                    column_boundaries.append((left_bound, right_bound))
                
            except:
                # Fallback: equal division
                column_width = (x_max - x_min) / num_columns
                column_boundaries = [
                    (x_min + i * column_width, x_min + (i + 1) * column_width) 
                    for i in range(num_columns)
                ]
        else:
            # Simple division for few points
            column_width = (x_max - x_min) / num_columns if num_columns > 1 else (x_max - x_min)
            column_boundaries = [
                (x_min + i * column_width, x_min + (i + 1) * column_width) 
                for i in range(num_columns)
            ]
        
        # Group detections by column
        column_detections = [[] for _ in range(num_columns)]
        
        for detection in detections:
            x = detection['center_x']
            assigned = False
            
            for col_idx, (left_bound, right_bound) in enumerate(column_boundaries):
                if left_bound <= x <= right_bound:
                    column_detections[col_idx].append(detection)
                    assigned = True
                    break
            
            # Fallback: assign to nearest column
            if not assigned:
                nearest_col = min(range(num_columns), 
                                key=lambda i: min(abs(x - column_boundaries[i][0]), 
                                                abs(x - column_boundaries[i][1])))
                column_detections[nearest_col].append(detection)
        
        # Create spatial position mapping
        position_grid = {}
        column_info = {}
        
        for col_idx, col_detections in enumerate(column_detections):
            if not col_detections:
                continue
            
            # Sort by Y coordinate
            col_detections.sort(key=lambda x: x['center_y'])
            
            # Handle multiple detections per expected position
            y_coords_col = [d['center_y'] for d in col_detections]
            expected_rows = questions_per_column
            
            if len(col_detections) <= expected_rows:
                # Direct assignment
                for row_idx, detection in enumerate(col_detections):
                    question_num = col_idx * questions_per_column + row_idx + 1
                    if question_num <= num_questions:
                        position_grid[question_num] = detection
            else:
                # Use clustering to handle multiple detections per row
                try:
                    n_clusters = min(expected_rows, len(set(y_coords_col)))
                    if n_clusters > 1:
                        kmeans_y = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        y_clusters = kmeans_y.fit_predict(np.array(y_coords_col).reshape(-1, 1))
                        
                        # Group by clusters
                        cluster_groups = defaultdict(list)
                        for detection, cluster_id in zip(col_detections, y_clusters):
                            cluster_groups[cluster_id].append(detection)
                        
                        # Sort clusters by Y position and assign
                        sorted_clusters = sorted(cluster_groups.items(), 
                                               key=lambda x: np.mean([d['center_y'] for d in x[1]]))
                        
                        for row_idx, (cluster_id, cluster_detections) in enumerate(sorted_clusters):
                            if row_idx < expected_rows:
                                # Take detection with highest confidence
                                best_detection = max(cluster_detections, key=lambda x: x['confidence'])
                                question_num = col_idx * questions_per_column + row_idx + 1
                                if question_num <= num_questions:
                                    position_grid[question_num] = best_detection
                    else:
                        # Single cluster, take first detection
                        question_num = col_idx * questions_per_column + 1
                        if question_num <= num_questions:
                            position_grid[question_num] = col_detections[0]
                
                except:
                    # Fallback: take evenly distributed detections
                    step = max(1, len(col_detections) // expected_rows)
                    for row_idx in range(expected_rows):
                        detection_idx = min(row_idx * step, len(col_detections) - 1)
                        question_num = col_idx * questions_per_column + row_idx + 1
                        if question_num <= num_questions:
                            position_grid[question_num] = col_detections[detection_idx]
            
            column_info[col_idx] = {
                'detections_count': len(col_detections),
                'boundaries': column_boundaries[col_idx],
                'y_range': (min(y_coords_col), max(y_coords_col)) if y_coords_col else (0, 0)
            }
        
        return position_grid, column_info
    
    def validate_spatial_consistency(self, position_grid, num_questions=40, questions_per_column=10):
        """
        Validasi konsistensi spatial dan koreksi otomatis
        """
        validated_answers = {}
        corrections_made = []
        
        # Group by columns for validation
        questions_by_column = defaultdict(list)
        for q_num, detection in position_grid.items():
            col_idx = (q_num - 1) // questions_per_column
            questions_by_column[col_idx].append((q_num, detection))
        
        # Validate each column's spatial ordering
        for col_idx, col_questions in questions_by_column.items():
            # Sort by question number first
            col_questions.sort(key=lambda x: x[0])
            
            # Check if Y coordinates are in ascending order
            y_coords = [detection['center_y'] for _, detection in col_questions]
            
            # If not properly ordered, re-sort by Y coordinate
            if len(y_coords) > 1 and not all(y_coords[i] <= y_coords[i+1] for i in range(len(y_coords)-1)):
                corrections_made.append(f"Re-ordered column {col_idx} based on Y coordinates")
                
                # Re-sort by Y coordinate
                col_questions.sort(key=lambda x: x[1]['center_y'])
                
                # Reassign question numbers based on correct spatial order
                base_question = col_idx * questions_per_column + 1
                for i, (old_q_num, detection) in enumerate(col_questions):
                    new_q_num = base_question + i
                    if new_q_num <= num_questions:
                        validated_answers[new_q_num] = detection['class']
            else:
                # Already in correct order
                for q_num, detection in col_questions:
                    if q_num <= num_questions:
                        validated_answers[q_num] = detection['class']
        
        if corrections_made:
            logger.info(f"Spatial corrections applied: {'; '.join(corrections_made)}")
        
        return validated_answers
    
    def extract_answers_with_layout_analysis(self, detections, num_questions=40, questions_per_column=10):
        """
        Extract answers dengan layout analysis yang tidak terpengaruh sequential error
        Enhanced version yang menggabungkan spatial mapping dengan backward compatibility
        """
        if not detections:
            logger.warning("No detections found")
            return {}
        
        # Filter overlapping detections untuk menghindari duplikasi
        filtered_detections = self.filter_overlapping_detections(detections)
        logger.debug(f"After filtering overlaps: {len(filtered_detections)} detections remain")
        
        # Create spatial grid mapping (NEW: Anti-sequential error)
        position_grid, column_info = self.create_spatial_grid(
            filtered_detections, num_questions, questions_per_column
        )
        
        logger.debug(f"Spatial grid created with {len(position_grid)} position mappings")
        
        # Validate and correct spatial consistency
        validated_answers = self.validate_spatial_consistency(
            position_grid, num_questions, questions_per_column
        )
        
        logger.debug(f"Extracted answers for {len(validated_answers)} questions with spatial validation")
        
        return validated_answers
    
    def process_answer_sheet(self, image_path, confidence_threshold=0.5, num_questions=40, questions_per_column=10):
        """
        Method utama untuk memproses lembar jawaban dengan hasil yang konsisten
        """
        try:
            # Deteksi answers
            detections, processed_image = self.detect_answers(image_path, confidence_threshold)
            
            # Extract answers dengan enhanced layout analysis
            answers = self.extract_answers_with_layout_analysis(
                detections, 
                num_questions=num_questions, 
                questions_per_column=questions_per_column
            )
            
            logger.info(f"Successfully processed {len(answers)} answers from {image_path}")
            return answers
            
        except Exception as e:
            logger.error(f"Error processing answer sheet {image_path}: {str(e)}")
            return {}
    
    def debug_spatial_analysis(self, image_path, confidence_threshold=0.5, num_questions=40, questions_per_column=10):
        """
        Method debug untuk menganalisis spatial mapping (optional - untuk troubleshooting)
        """
        try:
            detections, processed_image = self.detect_answers(image_path, confidence_threshold)
            
            print(f"=== SPATIAL ANALYSIS DEBUG ===")
            print(f"Total raw detections: {len(detections)}")
            
            # Filter overlaps
            filtered_detections = self.filter_overlapping_detections(detections)
            print(f"After filtering overlaps: {len(filtered_detections)}")
            
            # Create spatial grid
            position_grid, column_info = self.create_spatial_grid(
                filtered_detections, num_questions, questions_per_column
            )
            
            print(f"\n=== COLUMN ANALYSIS ===")
            for col_idx, info in column_info.items():
                print(f"Column {col_idx+1}: {info['detections_count']} detections")
                print(f"  X boundaries: ({info['boundaries'][0]:.1f}, {info['boundaries'][1]:.1f})")
                print(f"  Y range: ({info['y_range'][0]:.1f}, {info['y_range'][1]:.1f})")
            
            # Validate and get final answers
            validated_answers = self.validate_spatial_consistency(
                position_grid, num_questions, questions_per_column
            )
            
            print(f"\n=== RESULTS ===")
            print(f"Spatial mappings: {len(position_grid)}")
            print(f"Final validated answers: {len(validated_answers)}")
            
            missing = set(range(1, num_questions + 1)) - set(validated_answers.keys())
            if missing:
                print(f"Missing questions: {sorted(missing)}")
            
            return validated_answers
            
        except Exception as e:
            logger.error(f"Error in debug spatial analysis: {str(e)}")
            return {}
        
class QuizCorrectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Koreksi Jawaban Pilihan Ganda")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Data storage
        self.answer_key = [""] * 40  # Kunci jawaban 40 soal
        self.detected_answers = []
        
        # Initialize detector
        self.detector = None
        self.model_path = ""
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Show welcome page
        self.show_welcome_page()
    
    def clear_frame(self):
        """Clear all widgets from main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_welcome_page(self):
        """Halaman 1: Halaman Awal"""
        self.clear_frame()
        
        title_label = tk.Label(
            self.main_frame, 
            text="Selamat Datang", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(pady=30)
        
        subtitle_label = tk.Label(
            self.main_frame, 
            text="Aplikasi Koreksi Jawaban Pilihan Ganda", 
            font=("Arial", 16),
            bg="#f0f0f0",
            fg="#34495e"
        )
        subtitle_label.pack(pady=10)
        
        desc_label = tk.Label(
            self.main_frame, 
            text="Sistem untuk mengoreksi lembar jawaban pilihan ganda 40 soal\ndengan teknologi deteksi gambar otomatis", 
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#7f8c8d",
            justify="center"
        )
        desc_label.pack(pady=20)
        
        model_frame = tk.LabelFrame(self.main_frame, text="Load YOLO Model", 
                                  font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        model_frame.pack(fill="x", padx=50, pady=20)
        
        load_model_btn = tk.Button(
            model_frame,
            text="Browse Model File (.pt)",
            font=("Arial", 11),
            bg="#9b59b6",
            fg="white",
            command=self.load_model_for_ui,
            cursor="hand2"
        )
        load_model_btn.pack(pady=15)
        
        self.model_status = tk.Label(
            model_frame,
            text="Model belum dimuat",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#e74c3c"
        )
        self.model_status.pack(pady=5)
        
        self.start_btn = tk.Button(
            self.main_frame,
            text="Mulai Koreksi",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            pady=15,
            padx=30,
            command=self.show_answer_key_page,
            cursor="hand2",
            state="disabled"
        )
        self.start_btn.pack(pady=30)
        
        info_frame = tk.Frame(self.main_frame, bg="#ecf0f1", relief="ridge", bd=2)
        info_frame.pack(pady=20, padx=50, fill="x")
        
        info_title = tk.Label(
            info_frame,
            text="Langkah-langkah:",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        info_title.pack(pady=10)
        
        steps = [
            "1. Input kunci jawaban",
            "2. Koreksi lembar jawaban siswa",
            "3. Lihat hasil koreksi"
        ]
        
        for step in steps:
            step_label = tk.Label(
                info_frame,
                text=step,
                font=("Arial", 10),
                bg="#ecf0f1",
                fg="#34495e"
            )
            step_label.pack(anchor="w", padx=20, pady=2)
        
        info_frame.pack_configure(pady=(20, 10))
    
    def load_model_for_ui(self):
        """Load YOLO model - UI version"""
        file_path = filedialog.askopenfilename(
            title="Pilih File Model YOLO",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        
        if file_path:
            try:
                self.model_status.config(text="Memuat model...", fg="#f39c12")
                self.root.update()
                
                self.detector = AnswerSheetDetector(file_path)
                self.model_path = file_path
                
                if self.detector.is_loaded:
                    self.model_status.config(text=f"Model dimuat: {os.path.basename(file_path)}", fg="#27ae60")
                    self.start_btn.config(state="normal")
                    messagebox.showinfo("Berhasil", "Model YOLO berhasil dimuat!")
                    logger.debug(f"Model loaded in UI: {file_path}")
                    # Warn if model differs from expected
                    expected_model = "D:/Developments/KP/GANN/Hasil Training Model Yolov11n/best.pt"
                    if file_path != expected_model:
                        logger.warning(f"Selected model {file_path} differs from expected {expected_model}")
                else:
                    raise Exception("Failed to load model")
                
            except Exception as e:
                self.model_status.config(text="Gagal memuat model", fg="#e74c3c")
                messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")
                logger.error(f"Model loading error in UI: {str(e)}")
    
    def show_answer_key_page(self):
        """Halaman 2: Input Kunci Jawaban dengan layout kiri-kanan"""
        self.clear_frame()
        
        header_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = tk.Label(
            header_frame, 
            text="Input Kunci Jawaban", 
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(side="left")
        
        next_btn = tk.Button(
            header_frame,
            text="Lanjut ke Koreksi",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            command=self.show_correction_page,
            cursor="hand2",
            padx=20,
            pady=8
        )
        next_btn.pack(side="right")
        
        main_container = tk.Frame(self.main_frame, bg="#f0f0f0")
        main_container.pack(fill="both", expand=True, pady=(0, 20))
        
        left_frame = tk.Frame(main_container, bg="#f0f0f0")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        upload_frame = tk.LabelFrame(left_frame, text="Upload Lembar Kunci Jawaban", 
                                   font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        upload_frame.pack(fill="x", pady=(0, 20))
        
        upload_btn = tk.Button(
            upload_frame,
            text="Browse Gambar Kunci Jawaban",
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            command=self.upload_answer_key,
            cursor="hand2"
        )
        upload_btn.pack(pady=15, padx=20)
        
        self.upload_status = tk.Label(
            upload_frame,
            text="Belum ada file yang dipilih",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#7f8c8d"
        )
        self.upload_status.pack(pady=5)
        
        manual_frame = tk.LabelFrame(left_frame, text="Input Manual Kunci Jawaban", 
                                   font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        manual_frame.pack(fill="both", expand=True)
        
        self.answer_entries = []
        grid_frame = tk.Frame(manual_frame, bg="#f0f0f0")
        grid_frame.pack(pady=15)
        
        for i in range(8):
            for j in range(5):
                question_num = i * 5 + j + 1
                if question_num <= 40:
                    q_frame = tk.Frame(grid_frame, bg="#f0f0f0")
                    q_frame.grid(row=i, column=j, padx=3, pady=3)
                    
                    tk.Label(q_frame, text=f"{question_num}:", 
                           font=("Arial", 8), bg="#f0f0f0").pack()
                    
                    entry = tk.Entry(q_frame, width=3, font=("Arial", 10), 
                                   justify="center", bd=1, relief="solid")
                    entry.pack()
                    entry.bind('<KeyRelease>', lambda e, idx=question_num-1: self.validate_answer(e, idx))
                    self.answer_entries.append(entry)
        
        right_frame = tk.Frame(main_container, bg="#f0f0f0")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        preview_frame = tk.LabelFrame(right_frame, text="Preview Kunci Jawaban", 
                                    font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        preview_frame.pack(fill="both", expand=True)
        
        self.preview_grid = tk.Frame(preview_frame, bg="#f0f0f0")
        self.preview_grid.pack(pady=15)
        
        self.preview_labels = []
        for i in range(8):
            for j in range(5):
                question_num = i * 5 + j + 1
                if question_num <= 40:
                    answer_frame = tk.Frame(self.preview_grid, bg="#ecf0f1", relief="ridge", bd=1, width=50, height=40)
                    answer_frame.grid(row=i, column=j, padx=2, pady=2, sticky="ew")
                    answer_frame.grid_propagate(False)
                    
                    tk.Label(answer_frame, text=f"{question_num}:", 
                           font=("Arial", 7, "bold"), bg="#ecf0f1").pack(pady=(2,0))
                    
                    answer_label = tk.Label(answer_frame, text="-", 
                           font=("Arial", 11, "bold"), bg="#ecf0f1", 
                           fg="#e74c3c")
                    answer_label.pack()
                    self.preview_labels.append(answer_label)
        
        self.summary_label = tk.Label(
            preview_frame,
            text="Kunci jawaban belum lengkap: 0/40",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#7f8c8d"
        )
        self.summary_label.pack(side="bottom", pady=10)
        
        back_btn = tk.Button(
            self.main_frame,
            text="Kembali",
            font=("Arial", 12, "bold"),
            bg="#95a5a6",
            fg="white",
            command=self.show_welcome_page,
            cursor="hand2",
            padx=20,
            pady=10
        )
        back_btn.pack(side="left", pady=10)
    
    def validate_answer(self, event, index):
        """Validate answer input (A, B, C, D only) and update preview"""
        entry = event.widget
        value = entry.get().upper()
        
        if value and value not in ['A', 'B', 'C', 'D']:
            entry.delete(0, tk.END)
            messagebox.showwarning("Input Tidak Valid", "Jawaban harus A, B, C, atau D")
        else:
            self.answer_key[index] = value
            self.update_preview()
    
    def update_preview(self):
        """Update preview display"""
        filled_count = 0
        for i, answer in enumerate(self.answer_key):
            if i < len(self.preview_labels):
                if answer:
                    self.preview_labels[i].config(text=answer, fg="#27ae60")
                    filled_count += 1
                else:
                    self.preview_labels[i].config(text="-", fg="#e74c3c")
        
        if filled_count == 40:
            self.summary_label.config(text="Kunci jawaban lengkap: 40/40", fg="#27ae60")
        else:
            self.summary_label.config(text=f"Kunci jawaban belum lengkap: {filled_count}/40", fg="#e74c3c")
    
    def upload_answer_key(self):
        """Upload and detect answer key from image"""
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Kunci Jawaban",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.upload_status.config(text=f"File: {os.path.basename(file_path)}")
            logger.debug(f"Selected image for answer key: {file_path}")
            try:
                img = cv2.imread(file_path)
                logger.debug(f"Image details: shape={img.shape}, format={os.path.splitext(file_path)[1]}")
            except Exception as e:
                logger.error(f"Invalid image: {str(e)}")
                messagebox.showerror("Error", "Gambar tidak valid")
                return
            self.detect_answers_from_image(file_path)
    
    def detect_answers_from_image(self, image_path):
        """Detect answers from uploaded image"""
        def process():
            try:
                self.root.after(0, lambda: self.upload_status.config(text="Memproses gambar...", fg="#f39c12"))
                answers = self.detector.process_answer_sheet(
                    image_path, confidence_threshold=0.3, num_questions=40, questions_per_column=10
                )
                
                detected_answers = [""] * 40
                detected_count = 0
                for q_num, answer in answers.items():
                    if 1 <= q_num <= 40:
                        detected_answers[q_num - 1] = answer
                        detected_count += 1
                
                empty_count = 40 - detected_count
                logger.debug(f"Detection result: detected={detected_count}, answers={detected_answers}")
                
                if detected_count > 0:
                    self.root.after(0, self.update_detected_answers_ui, 
                                   detected_answers, detected_count, empty_count)
                else:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Deteksi Gagal", "Tidak ada jawaban yang terdeteksi. Silakan periksa gambar dan coba lagi."))
                    self.root.after(0, lambda: self.upload_status.config(text="Gagal mendeteksi jawaban", fg="#e74c3c"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}"))
                self.root.after(0, lambda: self.upload_status.config(text="Gagal mendeteksi jawaban", fg="#e74c3c"))
                logger.error(f"Detection error: {str(e)}")
        
        if not self.detector or not self.detector.is_loaded:
            messagebox.showerror("Error", "Model YOLO belum dimuat. Silakan muat model terlebih dahulu.")
            self.upload_status.config(text="Model belum dimuat", fg="#e74c3c")
            logger.error("Detection attempted without loaded model")
            return
        
        self.last_image_path = image_path
        threading.Thread(target=process, daemon=True).start()
    
    def update_detected_answers_ui(self, detected_answers, detected_count, empty_count):
        """Update UI with detected answers"""
        try:
            logger.debug(f"Updating UI with detected answers: {detected_answers}")
            self.answer_key = detected_answers.copy()
            
            for i, answer in enumerate(detected_answers):
                if i < len(self.answer_entries):
                    self.answer_entries[i].delete(0, tk.END)
                    if answer:
                        self.answer_entries[i].insert(0, answer)
            
            self.update_preview()
            
            self.upload_status.config(
                text=f"Deteksi selesai: {detected_count} jawaban terdeteksi, {empty_count} kosong",
                fg="#27ae60" if detected_count > 0 else "#e74c3c"
            )
            
            if detected_count > 0:
                messagebox.showinfo("Deteksi Berhasil", 
                                  f"Berhasil mendeteksi {detected_count} jawaban dari 40 soal!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error updating UI: {str(e)}")
            logger.error(f"UI update error: {str(e)}")
    
    def show_correction_page(self):
        """Halaman 3: Koreksi Lembar Jawaban Siswa"""
        if not all(self.answer_key):
            messagebox.showwarning("Kunci Jawaban Belum Lengkap", 
                                 "Silakan lengkapi semua kunci jawaban sebelum melanjutkan!")
            return
        
        self.clear_frame()
        
        title_label = tk.Label(
            self.main_frame, 
            text="Koreksi Lembar Jawaban Siswa", 
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(pady=20)
        
        upload_frame = tk.LabelFrame(self.main_frame, text="Upload Lembar Jawaban Siswa", 
                              font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        upload_frame.pack(fill="x", padx=20, pady=10)
        
        upload_btn = tk.Button(
            upload_frame,
            text="Browse Lembar Jawaban Siswa",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            command=self.upload_student_answer,
            cursor="hand2",
            pady=10
        )
        upload_btn.pack(pady=15)
        
        self.student_file_status = tk.Label(
            upload_frame,
            text="Belum ada file yang dipilih",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#555"
        )
        self.student_file_status.pack(pady=5)
        
        self.results_frame = tk.LabelFrame(self.main_frame, text="Hasil Koreksi", 
                                     font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#333")
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.results_content = tk.Frame(self.results_frame, bg="#f0f0f0")
        
        btn_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        back_btn = tk.Button(
            btn_frame,
            text="Kembali ke Input",
            font=("Arial", 11),
            bg="#555555",
            fg="white",
            command=self.show_answer_key_page,
            cursor="hand2"
        )
        back_btn.pack(side="left", padx=10)
        
        new_correction_btn = tk.Button(
            btn_frame,
            text="Koreksi Baru",
            font=("Arial", 11),
            bg="#e67e22",
            fg="white",
            command=self.show_welcome_page,
            cursor="hand2"
        )
        new_correction_btn.pack(side="right", padx=10)
    
    def upload_student_answer(self):
        """Upload student answer sheet"""
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Lembar Jawaban Siswa",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.student_file_status.config(text=f"File: {os.path.basename(file_path)}")
            logger.debug(f"Logged student answer sheet: {file_path}")
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Cannot read image file")
                logger.debug(f"Image details: shape={img.shape}, format={os.path.splitext(file_path)[1]}")
            except Exception as e:
                logger.error(f"Invalid image: {str(e)}")
                messagebox.showerror("Error", "Gambar tidak valid")
                return
            self.process_student_answer(file_path)


    def process_student_answer(self, image_path):
        """Process and correct student answer sheet"""
        def process():
            # Mulai hitung waktu eksekusi
            start_time = time.time()
            
            try:
                # Update status
                self.root.after(0, lambda: self.student_file_status.config(text="Memproses gambar...", fg="#f39c12"))
                
                # Validasi answer key
                if not hasattr(self, 'answer_key') or not self.answer_key:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Kunci jawaban belum di-upload"))
                    return
                
                # Deteksi jawaban siswa
                student_answers = self.detector.process_answer_sheet(
                    image_path, confidence_threshold=0.3, num_questions=40, questions_per_column=10
                )
                
                # Proses hasil deteksi
                detected_student_answers = [""] * 40
                detected_count = 0
                for q_num, answer in student_answers.items():
                    if 1 <= q_num <= 40:
                        detected_student_answers[q_num - 1] = answer
                        detected_count += 1
                
                # Koreksi jawaban
                correction_results = self.correct_answers(detected_student_answers)
                
                # Hitung waktu eksekusi
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Update UI dengan hasil (termasuk waktu eksekusi)
                self.root.after(0, self.display_correction_results, 
                            correction_results, detected_student_answers, detected_count, execution_time)
                
            except Exception as e:
                # Hitung waktu meskipun ada error
                execution_time = time.time() - start_time
                self.root.after(0, lambda: messagebox.showerror("Error", f"Gagal memproses jawaban siswa: {str(e)}"))
                self.root.after(0, lambda: self.student_file_status.config(text="Gagal memproses jawaban", fg="#e74c3c"))
                logger.error(f"Student answer processing error after {execution_time:.2f} seconds: {str(e)}")
        
        # Validasi detector
        if not self.detector or not self.detector.is_loaded:
            messagebox.showerror("Error", "Model YOLO belum dimuat. Silakan muat model terlebih dahulu.")
            self.student_file_status.config(text="Model belum dimuat", fg="#e74c3c")
            logger.error("Student answer processing attempted without loaded model")
            return
        
        # Jalankan proses di thread terpisah
        threading.Thread(target=process, daemon=True).start()

    def correct_answers(self, student_answers):
        """Compare student answers with answer key and calculate score"""
        results = {
            'correct': 0,
            'wrong': 0,
            'empty': 0,
            'details': [],
            'score': 0,
            'percentage': 0
        }
        
        for i in range(40):
            question_num = i + 1
            correct_answer = self.answer_key[i]
            student_answer = student_answers[i] if i < len(student_answers) else ""
            
            detail = {
                'question': question_num,
                'correct_answer': correct_answer,
                'student_answer': student_answer,
                'status': ''
            }
            
            if not student_answer:
                detail['status'] = 'empty'
                results['empty'] += 1
            elif student_answer == correct_answer:
                detail['status'] = 'correct'
                results['correct'] += 1
            else:
                detail['status'] = 'wrong'
                results['wrong'] += 1
            
            results['details'].append(detail)
        
        # Hitung skor
        results['score'] = results['correct']
        results['percentage'] = (results['correct'] / 40) * 100
        
        logger.debug(f"Correction results: {results['correct']} correct, {results['wrong']} wrong, {results['empty']} empty")
        return results

    def display_correction_results(self, results, student_answers, detected_count, execution_time=None):
        """Display correction results in UI"""
        # Clear previous results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        self.results_content.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Status update - tambahkan waktu eksekusi
        status_text = f"Selesai: {detected_count} jawaban terdeteksi, skor: {results['score']}/40"
        if execution_time is not None:
            status_text += f" | Waktu: {execution_time:.2f} detik"
        
        self.student_file_status.config(
            text=status_text,
            fg="#27ae60"
        )
        
        # Summary frame
        summary_frame = tk.Frame(self.results_content, bg="#ecf0f1", relief="ridge", bd=2)
        summary_frame.pack(fill="x", pady=(0, 15))
        
        summary_title = tk.Label(
            summary_frame,
            text="Ringkasan Hasil Koreksi",
            font=("Arial", 14, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        summary_title.pack(pady=10)
        
        summary_info = tk.Frame(summary_frame, bg="#ecf0f1")
        summary_info.pack(pady=5)
        
        # Score info
        score_frame = tk.Frame(summary_info, bg="#ecf0f1")
        score_frame.pack(side="left", padx=20)
        
        tk.Label(score_frame, text="SKOR AKHIR", font=("Arial", 10, "bold"), 
                bg="#ecf0f1", fg="#2c3e50").pack()
        tk.Label(score_frame, text=f"{results['score']}/40", 
                font=("Arial", 20, "bold"), bg="#ecf0f1", fg="#e74c3c").pack()
        tk.Label(score_frame, text=f"({results['percentage']:.1f}%)", 
                font=("Arial", 12), bg="#ecf0f1", fg="#7f8c8d").pack()
        
        # Statistics
        stats_frame = tk.Frame(summary_info, bg="#ecf0f1")
        stats_frame.pack(side="right", padx=20)
        
        tk.Label(stats_frame, text=f"Benar: {results['correct']}", 
                font=("Arial", 10), bg="#ecf0f1", fg="#27ae60").pack(anchor="w")
        tk.Label(stats_frame, text=f"Salah: {results['wrong']}", 
                font=("Arial", 10), bg="#ecf0f1", fg="#e74c3c").pack(anchor="w")
        tk.Label(stats_frame, text=f"Kosong: {results['empty']}", 
                font=("Arial", 10), bg="#ecf0f1", fg="#f39c12").pack(anchor="w")
        tk.Label(stats_frame, text=f"Terdeteksi: {detected_count}/40", 
                font=("Arial", 10), bg="#ecf0f1", fg="#3498db").pack(anchor="w")
        
        # Tambahkan informasi waktu eksekusi jika ada
        if execution_time is not None:
            tk.Label(stats_frame, text=f"Waktu Proses: {execution_time:.2f} detik", 
                    font=("Arial", 10, "bold"), bg="#ecf0f1", fg="#9b59b6").pack(anchor="w")
        
        # Detailed results with scrollbar
        details_frame = tk.LabelFrame(self.results_content, text="Detail Koreksi", 
                                    font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#2c3e50")
        details_frame.pack(fill="both", expand=True, pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(details_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Headers
        headers_frame = tk.Frame(scrollable_frame, bg="#34495e")
        headers_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(headers_frame, text="No", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=5).pack(side="left", padx=2)
        tk.Label(headers_frame, text="Kunci", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=8).pack(side="left", padx=2)
        tk.Label(headers_frame, text="Jawaban", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=8).pack(side="left", padx=2)
        tk.Label(headers_frame, text="Status", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=10).pack(side="left", padx=2)
        
        # Detail rows
        for i, detail in enumerate(results['details']):
            row_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            
            row_frame = tk.Frame(scrollable_frame, bg=row_color, relief="ridge", bd=1)
            row_frame.pack(fill="x", padx=5, pady=1)
            
            tk.Label(row_frame, text=str(detail['question']), 
                    font=("Arial", 9), bg=row_color, width=5).pack(side="left", padx=2)
            tk.Label(row_frame, text=detail['correct_answer'], 
                    font=("Arial", 9, "bold"), bg=row_color, width=8).pack(side="left", padx=2)
            
            student_answer_text = detail['student_answer'] if detail['student_answer'] else "-"
            tk.Label(row_frame, text=student_answer_text, 
                    font=("Arial", 9), bg=row_color, width=8).pack(side="left", padx=2)
            
            # Status with color coding
            status_colors = {
                'correct': '#27ae60',
                'wrong': '#e74c3c',
                'empty': '#f39c12'
            }
            status_texts = {
                'correct': '✓ BENAR',
                'wrong': '✗ SALAH',
                'empty': '- KOSONG'
            }
            
            status_color = status_colors.get(detail['status'], '#7f8c8d')
            status_text = status_texts.get(detail['status'], 'UNKNOWN')
            
            tk.Label(row_frame, text=status_text, 
                    font=("Arial", 9, "bold"), bg=row_color, 
                    fg=status_color, width=10).pack(side="left", padx=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Export button
        export_frame = tk.Frame(self.results_content, bg="#f0f0f0")
        export_frame.pack(fill="x", pady=10)
        
        export_btn = tk.Button(
            export_frame,
            text="Export Hasil ke TXT",
            font=("Arial", 11),
            bg="#16a085",
            fg="white",
            command=lambda: self.export_results(results, student_answers),
            cursor="hand2",
            padx=20
        )
        export_btn.pack(side="right")
        
        # Show success message - tambahkan waktu eksekusi
        success_message = f"Koreksi berhasil!\nSkor: {results['score']}/40 ({results['percentage']:.1f}%)"
        if execution_time is not None:
            success_message += f"\nWaktu proses: {execution_time:.2f} detik"
        
        messagebox.showinfo("Koreksi Selesai", success_message)

    def export_results(self, results, student_answers):
        """Export correction results to text file"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Simpan Hasil Koreksi",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 50 + "\n")
                    f.write("HASIL KOREKSI JAWABAN PILIHAN GANDA\n")
                    f.write("=" * 50 + "\n\n")
                    
                    f.write("RINGKASAN:\n")
                    f.write(f"Skor Akhir: {results['score']}/40 ({results['percentage']:.1f}%)\n")
                    f.write(f"Jawaban Benar: {results['correct']}\n")
                    f.write(f"Jawaban Salah: {results['wrong']}\n")
                    f.write(f"Jawaban Kosong: {results['empty']}\n\n")
                    
                    f.write("DETAIL KOREKSI:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{'No':<4} {'Kunci':<8} {'Jawaban':<8} {'Status':<10}\n")
                    f.write("-" * 50 + "\n")
                    
                    for detail in results['details']:
                        student_answer = detail['student_answer'] if detail['student_answer'] else "-"
                        status_text = {
                            'correct': 'BENAR',
                            'wrong': 'SALAH',
                            'empty': 'KOSONG'
                        }.get(detail['status'], 'UNKNOWN')
                        
                        f.write(f"{detail['question']:<4} {detail['correct_answer']:<8} {student_answer:<8} {status_text:<10}\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("Generated by Quiz Correction App\n")
                
                messagebox.showinfo("Export Berhasil", f"Hasil koreksi berhasil disimpan ke:\n{file_path}")
                logger.info(f"Results exported to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Export Gagal", f"Gagal menyimpan file: {str(e)}")
            logger.error(f"Export error: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = QuizCorrectionApp(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()