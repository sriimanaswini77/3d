#!/usr/bin/env python3
"""
Image Analysis GUI Application (Scikit-Image Watershed Approach)
Tab 1: ROI Selection with Polygon Close-on-First-Click
Tab 2: Step-by-Step Watershed Segmentation using Sobel Gradient & Histogram Markers
Tab 3: Batch Image Processing with Defect Classification
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np 
from scipy import ndimage as ndi
import json
import math
import os
from pathlib import Path


class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADDITIVE MANUFACTURING - DEFECT DETECTION")
        self.root.geometry("1400x900")

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=(10, 5))

        # Initialize tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text='ROI Selection')
        self.notebook.add(self.tab2, text='Watershed Analysis')
        self.notebook.add(self.tab3, text='Batch Processing')

        # Initialize Tab 1
        self.init_tab1()

        # Initialize Tab 2
        self.init_tab2()

        # Initialize Tab 3
        self.init_tab3()

        # Add footer label
        footer_frame = ttk.Frame(root)
        footer_frame.pack(side='bottom', fill='x', padx=10, pady=(0, 10))

        footer_label = ttk.Label(footer_frame, text="Designed and Developed by DEAIS - RCI",
                                font=('Arial', 10), anchor='center')
        footer_label.pack()

    # ==================== TAB 1: ROI SELECTION ====================

    def init_tab1(self):
        """Initialize Tab 1 - ROI Selection with Close-on-First-Click"""
        # Variables
        self.tab1_image = None
        self.tab1_cv_image = None
        self.tab1_photo = None
        self.tab1_polygon_points = []
        self.tab1_image_path = None
        self.tab1_polygon_closed = False

        # Control frame
        control_frame = ttk.Frame(self.tab1)
        control_frame.pack(side='top', fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Upload Image",
                   command=self.tab1_upload_image).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear Points",
                   command=self.tab1_clear_points).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Coordinates",
                   command=self.tab1_save_coordinates).pack(side='left', padx=5)

        # Info label
        self.tab1_info_label = ttk.Label(control_frame,
                                         text="Click on image to select polygon points (click on first point to close)")
        self.tab1_info_label.pack(side='left', padx=20)

        # Canvas frame
        canvas_frame = ttk.Frame(self.tab1)
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Canvas with scrollbars
        self.tab1_canvas = tk.Canvas(canvas_frame, bg='gray')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical',
                                    command=self.tab1_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal',
                                    command=self.tab1_canvas.xview)

        self.tab1_canvas.configure(yscrollcommand=v_scrollbar.set,
                                   xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        self.tab1_canvas.pack(side='left', fill='both', expand=True)

        # Bind mouse events
        self.tab1_canvas.bind('<Button-1>', self.tab1_canvas_click)
        # Mousewheel scrolling for better usability
        self.tab1_canvas.bind('<MouseWheel>', lambda e: self.tab1_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab1_canvas.bind('<Shift-MouseWheel>', lambda e: self.tab1_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab1_canvas.bind('<Control-MouseWheel>', self.tab1_zoom)
        self.tab1_canvas.bind('<plus>', lambda e: self.tab1_zoom_button(1.1))
        self.tab1_canvas.bind('<minus>', lambda e: self.tab1_zoom_button(0.9))
        self.tab1_canvas.bind('<equal>', lambda e: self.tab1_zoom_button(1.1))
        self.tab1_zoom_level = 1.0
        

    def tab1_upload_image(self):
        """Upload image for ROI selection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                       ("All files", "*.*")]
        )

        if file_path:
            self.tab1_image_path = file_path
            self.tab1_cv_image = cv2.imread(file_path)
            self.tab1_image = cv2.cvtColor(self.tab1_cv_image, cv2.COLOR_BGR2RGB)
            self.tab1_polygon_points = []
            self.tab1_polygon_closed = False
            self.tab1_display_image()

    
    def tab1_zoom(self, event):
        """Zoom image with Ctrl+MouseWheel"""
        if self.tab1_image is None:
            return
        
        factor = 1.1 if event.delta > 0 else 0.9
        self.tab1_zoom_level *= factor
        self.tab1_zoom_level = max(0.1, min(10.0, self.tab1_zoom_level))
        self.tab1_display_image()

    def tab1_zoom_button(self, factor):
        """Zoom image with +/- keys"""
        if self.tab1_image is None:
            return
        
        self.tab1_zoom_level *= factor
        self.tab1_zoom_level = max(0.1, min(10.0, self.tab1_zoom_level))
        self.tab1_display_image()
    
    def tab1_display_image(self):
        """Display image on canvas"""
        if self.tab1_image is not None:
            display_img = self.tab1_image.copy()

            if len(self.tab1_polygon_points) > 0:
                for i in range(len(self.tab1_polygon_points) - 1):
                    cv2.line(display_img, self.tab1_polygon_points[i],
                            self.tab1_polygon_points[i + 1], (0, 255, 0), 2)

                if self.tab1_polygon_closed:
                    cv2.line(display_img, self.tab1_polygon_points[-1],
                            self.tab1_polygon_points[0], (0, 255, 0), 2)

                for idx, point in enumerate(self.tab1_polygon_points):
                    cv2.circle(display_img, point, 7, (255, 0, 0), -1)
                    cv2.circle(display_img, point, 8, (255, 255, 255), 2)

            pil_img = Image.fromarray(display_img)
            new_width = int(pil_img.width * self.tab1_zoom_level)
            new_height = int(pil_img.height * self.tab1_zoom_level)
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tab1_photo = ImageTk.PhotoImage(pil_img)

            self.tab1_canvas.delete("all")
            self.tab1_canvas.create_image(0, 0, anchor='nw', image=self.tab1_photo)
            self.tab1_canvas.configure(scrollregion=self.tab1_canvas.bbox("all"))
            
            
    def tab1_canvas_click(self, event):
        """Handle canvas click to add polygon points or close polygon"""
        if self.tab1_image is not None and not self.tab1_polygon_closed:
            x = int(self.tab1_canvas.canvasx(event.x))
            y = int(self.tab1_canvas.canvasy(event.y))

            # Check if clicking near first point to close polygon
            if len(self.tab1_polygon_points) >= 3:
                first_point = self.tab1_polygon_points[0]
                distance = math.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)

                if distance <= 15:
                    self.tab1_polygon_closed = True
                    self.tab1_display_image()
                    self.tab1_info_label.config(
                        text=f"Polygon closed with {len(self.tab1_polygon_points)} points. Ready to save."
                    )
                    return

            # Add new point
            self.tab1_polygon_points.append((int(x), int(y)))
            self.tab1_display_image()

            if len(self.tab1_polygon_points) < 3:
                self.tab1_info_label.config(
                    text=f"Points selected: {len(self.tab1_polygon_points)} (need at least 3)"
                )
            else:
                self.tab1_info_label.config(
                    text=f"Points selected: {len(self.tab1_polygon_points)} (click on point 1 to close)"
                )

    def tab1_clear_points(self):
        """Clear all polygon points"""
        self.tab1_polygon_points = []
        self.tab1_polygon_closed = False
        self.tab1_display_image()
        self.tab1_info_label.config(
            text="Click on image to select polygon points (click on first point to close)"
        )

    def tab1_save_coordinates(self):
        """Save polygon coordinates to JSON file"""
        if len(self.tab1_polygon_points) < 3:
            messagebox.showwarning("Warning",
                                   "Please select at least 3 points to form a polygon")
            return

        if self.tab1_image_path is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Coordinates",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if save_path:
            data = {
                "image_path": self.tab1_image_path,
                "image_shape": self.tab1_image.shape[:2],
                "polygon_points": self.tab1_polygon_points
            }

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

            messagebox.showinfo("Success", f"Coordinates saved to {save_path}")

    # ==================== TAB 2: WATERSHED ANALYSIS ====================

    def init_tab2(self):
        """Initialize Tab 2 - Step-by-Step Watershed Analysis (Scikit-Image Approach)"""
        # Variables
        self.tab2_image = None
        self.tab2_cv_image = None
        self.tab2_roi_points = None
        self.tab2_image_path = None
        self.current_step = 0
        self.step_images = {}
        self.step_data = {}

        # Default parameters for each step
        self.params = {
            'step_2_grayscale': {'enabled': True},
            'step_3_sobel': {'enabled': True},
            'step_4_markers': {'enabled': True, 'low_threshold': 30, 'high_threshold': 150},
            'step_5_watershed': {'enabled': True},
            'step_6_morph_close': {'enabled': True, 'iterations': 2},
            'step_7_morph_open': {'enabled': True, 'iterations': 2},
            'step_8_fillholes': {'enabled': True},
            'step_9_labeling': {'enabled': True, 'min_size': 0},
            'step_10_final': {'min_area': 0, 'max_area': 10000}
        }

        # Main container
        main_container = ttk.Frame(self.tab2)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Top control frame
        top_control_frame = ttk.Frame(main_container)
        top_control_frame.pack(side='top', fill='x', pady=5)

        ttk.Button(top_control_frame, text="Upload Image",
                   command=self.tab2_upload_image).pack(side='left', padx=5)
        ttk.Button(top_control_frame, text="Load ROI Coordinates",
                   command=self.tab2_load_roi).pack(side='left', padx=5)
        ttk.Button(top_control_frame, text="Start Processing",
                   command=self.tab2_start_processing).pack(side='left', padx=5)
        ttk.Button(top_control_frame, text="Save Configuration",
                   command=self.tab2_save_config).pack(side='left', padx=5)
        ttk.Button(top_control_frame, text="Load Configuration",
                   command=self.tab2_load_config).pack(side='left', padx=5)

        # Info label
        self.tab2_info_label = ttk.Label(top_control_frame,
                                         text="Upload image and load ROI coordinates to begin")
        self.tab2_info_label.pack(side='left', padx=20)

        # Content area (image + controls)
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill='both', expand=True, pady=5)

        # Left side: Image display
        image_container = ttk.Frame(content_frame)
        image_container.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Canvas with scrollbars for image
        self.tab2_canvas = tk.Canvas(image_container, bg='gray')
        v_scrollbar = ttk.Scrollbar(image_container, orient='vertical',
                                    command=self.tab2_canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_container, orient='horizontal',
                                    command=self.tab2_canvas.xview)

        self.tab2_canvas.configure(yscrollcommand=v_scrollbar.set,
                                   xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        self.tab2_canvas.pack(side='left', fill='both', expand=True)

        # Mousewheel scrolling for better usability
        self.tab2_canvas.bind('<MouseWheel>', lambda e: self.tab2_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab2_canvas.bind('<Shift-MouseWheel>', lambda e: self.tab2_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab2_canvas.bind('<Control-MouseWheel>', self.tab2_zoom)
        self.tab2_canvas.bind('<plus>', lambda e: self.tab2_zoom_button(1.1))
        self.tab2_canvas.bind('<minus>', lambda e: self.tab2_zoom_button(0.9))
        self.tab2_canvas.bind('<equal>', lambda e: self.tab2_zoom_button(1.1))
        self.tab2_zoom_level = 1.0

        # Right side: Parameter controls
        control_container = ttk.Frame(content_frame, width=350)
        control_container.pack(side='right', fill='y')
        control_container.pack_propagate(False)

        # Step info label
        step_info_frame = ttk.LabelFrame(control_container, text="Current Step", padding=10)
        step_info_frame.pack(fill='x', padx=5, pady=5)

        self.step_name_label = ttk.Label(step_info_frame, text="Step 0: Original Image",
                                          font=('Arial', 11, 'bold'))
        self.step_name_label.pack()

        self.step_desc_label = ttk.Label(step_info_frame, text="Load an image to begin",
                                          wraplength=300, justify='left')
        self.step_desc_label.pack(pady=(5, 0))

        # Parameters frame (scrollable)
        params_label_frame = ttk.LabelFrame(control_container, text="Parameters", padding=10)
        params_label_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create canvas for scrollable parameters
        self.params_canvas = tk.Canvas(params_label_frame, highlightthickness=0)
        params_scrollbar = ttk.Scrollbar(params_label_frame, orient='vertical',
                                         command=self.params_canvas.yview)
        self.params_frame = ttk.Frame(self.params_canvas)

        self.params_canvas.configure(yscrollcommand=params_scrollbar.set)
        params_scrollbar.pack(side='right', fill='y')
        self.params_canvas.pack(side='left', fill='both', expand=True)

        self.params_canvas_window = self.params_canvas.create_window((0, 0), window=self.params_frame,
                                                                       anchor='nw')

        # Update scroll region when frame changes
        self.params_frame.bind('<Configure>',
                               lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox('all')))

        # Navigation buttons
        nav_frame = ttk.Frame(control_container)
        nav_frame.pack(fill='x', padx=5, pady=5)

        self.prev_button = ttk.Button(nav_frame, text="Previous",
                                       command=self.tab2_previous_step, state='disabled')
        self.prev_button.pack(side='left', fill='x', expand=True, padx=(0, 2))

        self.next_button = ttk.Button(nav_frame, text="Next",
                                       command=self.tab2_next_step, state='disabled')
        self.next_button.pack(side='right', fill='x', expand=True, padx=(2, 0))

        # Step progress
        progress_frame = ttk.Frame(control_container)
        progress_frame.pack(fill='x', padx=5, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Step 0 / 9")
        self.progress_label.pack()

        self.tab2_photo = None

    def tab2_upload_image(self):
        """Upload image for watershed analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                       ("All files", "*.*")]
        )

        if file_path:
            self.tab2_image_path = file_path
            self.tab2_cv_image = cv2.imread(file_path)
            self.tab2_image = cv2.cvtColor(self.tab2_cv_image, cv2.COLOR_BGR2RGB)

            # Display the image immediately
            self.step_images = {}
            self.step_images[0] = self.tab2_image.copy()
            self.current_step = 0
            self.display_current_step()

            self.tab2_info_label.config(text="Image loaded. Load ROI coordinates and click Start Processing.")

    def tab2_load_roi(self):
        """Load ROI coordinates from JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select ROI Coordinates File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.tab2_roi_points = np.array(data['polygon_points'], dtype=np.int32)

            # Load parameters if they exist in the file
            if 'pipeline_params' in data:
                self.params = data['pipeline_params']

            self.tab2_info_label.config(
                text=f"ROI loaded with {len(self.tab2_roi_points)} points. Click Start Processing."
            )

    def tab2_zoom(self, event):
        """Zoom image with Ctrl+MouseWheel"""
        if self.current_step not in self.step_images:
            return
        
        factor = 1.1 if event.delta > 0 else 0.9
        self.tab2_zoom_level *= factor
        self.tab2_zoom_level = max(0.1, min(10.0, self.tab2_zoom_level))
        self.display_current_step()

    def tab2_zoom_button(self, factor):
        """Zoom image with +/- keys"""
        if self.current_step not in self.step_images:
            return
        
        self.tab2_zoom_level *= factor
        self.tab2_zoom_level = max(0.1, min(10.0, self.tab2_zoom_level))
        self.display_current_step()
        
    def tab2_start_processing(self):
        """Start the step-by-step watershed processing"""
        if self.tab2_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return

        if self.tab2_roi_points is None:
            messagebox.showwarning("Warning", "Please load ROI coordinates first")
            return

        # Process all steps
        self.process_all_steps()

        # Navigate to first step
        self.current_step = 0
        self.display_current_step()

        # Enable navigation
        self.next_button.config(state='normal')

        self.tab2_info_label.config(text="Processing complete. Navigate through steps using Next/Previous buttons.")

    def process_all_steps(self):
        """Process all watershed steps using Scikit-Image approach (Sobel + Histogram Markers)"""
        self.step_images = {}
        self.step_data = {}

        # Step 0: Original Image
        self.step_images[0] = self.tab2_image.copy()

        # Step 1: AoI Masked Image
        mask = np.zeros(self.tab2_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.tab2_roi_points], 255)
        masked_image = self.tab2_cv_image.copy()
        masked_image[mask == 0] = [0, 0, 0]
        self.step_images[1] = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        self.step_data['mask'] = mask
        self.step_data['masked_bgr'] = masked_image

        # Step 2: Grayscale Conversion
        if self.params['step_2_grayscale']['enabled']:
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            self.step_images[2] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            self.step_images[2] = self.step_images[1].copy()
        self.step_data['gray'] = gray

        # Step 3: Sobel Gradient (Elevation Map) - KEY STEP FOR WATERSHED
        if self.params['step_3_sobel']['enabled']:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            elevation_map = np.hypot(sobel_x, sobel_y)
            # Normalize for display
            elevation_display = cv2.normalize(elevation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            elevation_map = gray.astype(np.float64)
            elevation_display = gray
        self.step_images[3] = cv2.cvtColor(elevation_display, cv2.COLOR_GRAY2RGB)
        self.step_data['elevation_map'] = elevation_map

        # Step 4: Histogram-based Markers (Background and Foreground with varying values)
        if self.params['step_4_markers']['enabled']:
            low_thresh = self.params['step_4_markers']['low_threshold']
            high_thresh = self.params['step_4_markers']['high_threshold']
            
            inv = 255 - gray
            background = cv2.GaussianBlur(inv, (0,0), sigmaX=50, sigmaY=50)
            gbImg = cv2.subtract(inv, background)
            _, gb = cv2.threshold(gbImg, 10, 255, cv2.THRESH_BINARY)

				
            # Use varying marker values based on intensity
            markers = np.zeros_like(gray, dtype=np.int32)
            # Background markers: use actual pixel values (low intensities)
            background_mask = gray < low_thresh
            markers[background_mask] = gb[background_mask] #- gray[background_mask].astype(np.int32)   # Offset by 1 to avoid 0
            
            

            # Foreground markers: use actual pixel values (high intensities)
            foreground_mask = gray > high_thresh
            markers[foreground_mask] = gray[foreground_mask].astype(np.int32) + 1  # Offset by 1 to avoid 0

            # Visualize markers (show the varying values)
            markers_display = np.zeros_like(gray)
            markers_display[background_mask] = gb[background_mask]  # Show original values for background
            markers_display[foreground_mask] = gray[foreground_mask]  # Show original values for foreground
        else:
            # Skip: no markers, will segment entire image
            markers = np.zeros_like(gray, dtype=np.int32)
            markers[gray > 0] = gray[gray > 0].astype(np.int32) + 1
            markers_display = gray.copy()

        self.step_images[4] = cv2.cvtColor(markers_display, cv2.COLOR_GRAY2RGB)
        self.step_data['markers'] = markers

        # Step 5: Watershed Application on Elevation Map
        if self.params['step_5_watershed']['enabled']:
            # Apply watershed
            elevation_map_uint8 = elevation_display
            elevation_map_color = cv2.cvtColor(elevation_map_uint8, cv2.COLOR_GRAY2BGR)
            markers_watershed = markers.copy()
            segmentation = cv2.watershed(elevation_map_color, markers_watershed)

            # Visualize segmentation - show all regions distinctly
            watershed_display = np.zeros_like(self.step_data['masked_bgr'])
            # Background (label 1) - dark gray
            watershed_display[segmentation == 1] = [50, 50, 50]
            # Foreground (label 2) - green
            watershed_display[segmentation == 2] = [0, 200, 0]
            # Boundaries - bright red
            watershed_display[segmentation == -1] = [255, 0, 0]
            # Unknown regions (label 0) - black
            watershed_display[segmentation == 0] = [0, 0, 0]
        else:
            segmentation = markers.copy()
            watershed_display = self.step_data['masked_bgr'].copy()

        self.step_images[5] = cv2.cvtColor(watershed_display, cv2.COLOR_BGR2RGB)
        self.step_data['segmentation'] = segmentation

        # Convert segmentation to binary for morphological operations
        # Foreground regions have marker values corresponding to high intensities (> high_threshold)
        if self.params['step_4_markers']['enabled']:
            high_thresh = self.params['step_4_markers']['high_threshold']
            # Identify foreground as regions with labels > high_threshold
            binary_seg = (segmentation > high_thresh).astype(np.uint8)
        else:
            # If markers were skipped, all non-zero regions are foreground
            binary_seg = (segmentation > 0).astype(np.uint8)

        # Step 6: Morphological Close
        if self.params['step_6_morph_close']['enabled']:
            iterations = self.params['step_6_morph_close']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(binary_seg, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            morph_close = binary_seg.copy()

        self.step_images[6] = cv2.cvtColor(morph_close * 255, cv2.COLOR_GRAY2RGB)
        self.step_data['morph_close'] = morph_close

        # Step 7: Morphological Open
        if self.params['step_7_morph_open']['enabled']:
            iterations = self.params['step_7_morph_open']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            morph_open = morph_close.copy()

        self.step_images[7] = cv2.cvtColor(morph_open * 255, cv2.COLOR_GRAY2RGB)
        self.step_data['morph_open'] = morph_open

        # Step 8: Fill Holes (Remove holes in segmented regions)
        if self.params['step_8_fillholes']['enabled']:
            filled = ndi.binary_fill_holes(morph_open).astype(np.uint8) * 255
        else:
            filled = morph_open * 255

        self.step_images[8] = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)
        self.step_data['filled'] = filled

        # Step 9: Label Connected Components
        if self.params['step_9_labeling']['enabled']:
            labeled_array, num_features = ndi.label(filled)

            # Remove small objects
            min_size = self.params['step_9_labeling']['min_size']
            sizes = np.bincount(labeled_array.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0  # Remove background
            cleaned_labels = mask_sizes[labeled_array]

            # Re-label after cleaning
            labeled_cleaned, num_cleaned = ndi.label(cleaned_labels)

            # Visualize labeled regions
            labeled_display = cv2.normalize(labeled_cleaned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            labeled_colored = cv2.applyColorMap(labeled_display, cv2.COLORMAP_JET)
        else:
            labeled_cleaned, num_cleaned = ndi.label(filled)
            labeled_display = cv2.normalize(labeled_cleaned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            labeled_colored = cv2.applyColorMap(labeled_display, cv2.COLORMAP_JET)

        self.step_images[9] = cv2.cvtColor(labeled_colored, cv2.COLOR_BGR2RGB)
        self.step_data['labeled'] = labeled_cleaned
        self.step_data['num_labels'] = num_cleaned

        # Step 10: Final Result with Colored Segments
        #result = self.step_data['masked_bgr'].copy()
        result = self.tab2_cv_image.copy()

        min_area = self.params['step_10_final']['min_area']
        max_area = self.params['step_10_final']['max_area']

        # Color segments with red
        segment_color = (0, 0, 255)  # Red in BGR

        # Color each labeled region
        for region_label in range(1, self.step_data['num_labels'] + 1):
            region_mask = (labeled_cleaned == region_label).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area <= area <= max_area:
                        # Color the segment
                        cv2.drawContours(result, [contour], -1, segment_color, -1)

        self.step_images[10] = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def display_current_step(self):
        """Display the current step's image and parameters"""
        if self.current_step not in self.step_images:
            return

        # Update step info
        step_names = [
            "Step 0: Original Image",
            "Step 1: AoI Masked Image",
            "Step 2: Grayscale Conversion",
            "Step 3: Sobel Gradient (Elevation Map)",
            "Step 4: Histogram Markers",
            "Step 5: Watershed Application",
            "Step 6: Morphological Close",
            "Step 7: Morphological Open",
            "Step 8: Fill Holes",
            "Step 9: Connected Component Labeling",
            "Step 10: Final Result"
        ]

        step_descriptions = [
            "Original loaded image without any processing.",
            "Image with Area of Interest applied. Regions outside the polygon are blacked out.",
            "Image converted to grayscale for processing.",
            "Sobel gradient (elevation map) computed. High gradient values form barriers between regions.",
            "Markers with varying values: background uses low intensities, foreground uses high intensities.",
            "Watershed algorithm floods from markers using elevation map. Boundaries marked in red.",
            "Morphological closing to fill small holes and connect nearby objects.",
            "Morphological opening to remove small noise and separate touching objects.",
            "Binary fill holes removes small holes in detected foreground regions.",
            "Connected component labeling identifies individual objects. Small objects filtered out.",
            "Final result with detected segments colored in red."
        ]

        self.step_name_label.config(text=step_names[self.current_step])
        self.step_desc_label.config(text=step_descriptions[self.current_step])
        self.progress_label.config(text=f"Step {self.current_step} / 10")

        # Display image
        image = self.step_images[self.current_step]
        pil_img = Image.fromarray(image)
        new_width = int(pil_img.width * self.tab2_zoom_level)
        new_height = int(pil_img.height * self.tab2_zoom_level)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tab2_photo = ImageTk.PhotoImage(pil_img)

        self.tab2_canvas.delete("all")
        self.tab2_canvas.create_image(0, 0, anchor='nw', image=self.tab2_photo)
        self.tab2_canvas.configure(scrollregion=self.tab2_canvas.bbox("all"))

        # Update parameter controls
        self.update_parameter_controls()

        # Update navigation buttons
        self.prev_button.config(state='normal' if self.current_step > 0 else 'disabled')
        self.next_button.config(state='normal' if self.current_step < 10 else 'disabled')

    def update_parameter_controls(self):
        """Update parameter controls based on current step"""
        # Clear existing controls
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        step = self.current_step

        if step == 2:  # Grayscale
            self.create_step2_controls()
        elif step == 3:  # Sobel
            self.create_step3_controls()
        elif step == 4:  # Markers
            self.create_step4_controls()
        elif step == 5:  # Watershed
            self.create_step5_controls()
        elif step == 6:  # Morphological Close
            self.create_step6_controls()
        elif step == 7:  # Morphological Open
            self.create_step7_controls()
        elif step == 8:  # Fill Holes
            self.create_step8_controls()
        elif step == 9:  # Labeling
            self.create_step9_controls()
        elif step == 10:  # Final
            self.create_step10_controls()
        else:
            ttk.Label(self.params_frame, text="No adjustable parameters for this step.",
                     wraplength=280).pack(pady=10)

    def create_step2_controls(self):
        """Create controls for Step 2: Grayscale Conversion"""
        skip_var = tk.BooleanVar(value=not self.params['step_2_grayscale']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_2_grayscale', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Skipping shows RGB but processes grayscale internally.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step3_controls(self):
        """Create controls for Step 3: Sobel Gradient"""
        skip_var = tk.BooleanVar(value=not self.params['step_3_sobel']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_3_sobel', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Sobel computes image gradient as elevation map for watershed. High gradients form barriers between regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step4_controls(self):
        """Create controls for Step 4: Histogram Markers"""
        skip_var = tk.BooleanVar(value=not self.params['step_4_markers']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_4_markers', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        # Low threshold control
        ttk.Label(self.params_frame, text="Low Threshold (Background):").pack(anchor='w', pady=(5, 2))
        low_frame = ttk.Frame(self.params_frame)
        low_frame.pack(fill='x', pady=2)

        low_var = tk.IntVar(value=self.params['step_4_markers']['low_threshold'])
        low_scale = ttk.Scale(low_frame, from_=0, to=255, orient='horizontal',
                             variable=low_var,
                             command=lambda v: self.update_step4_params(low_var, None))
        low_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        low_spinbox = ttk.Spinbox(low_frame, from_=0, to=255, textvariable=low_var,
                                  width=6, command=lambda: self.update_step4_params(low_var, None))
        low_spinbox.pack(side='right')
        low_spinbox.bind('<Return>', lambda e: self.update_step4_params(low_var, None))
        low_spinbox.bind('<FocusOut>', lambda e: self.update_step4_params(low_var, None))

        # High threshold control
        ttk.Label(self.params_frame, text="High Threshold (Foreground):").pack(anchor='w', pady=(10, 2))
        high_frame = ttk.Frame(self.params_frame)
        high_frame.pack(fill='x', pady=2)

        high_var = tk.IntVar(value=self.params['step_4_markers']['high_threshold'])
        high_scale = ttk.Scale(high_frame, from_=0, to=255, orient='horizontal',
                              variable=high_var,
                              command=lambda v: self.update_step4_params(None, high_var))
        high_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        high_spinbox = ttk.Spinbox(high_frame, from_=0, to=255, textvariable=high_var,
                                   width=6, command=lambda: self.update_step4_params(None, high_var))
        high_spinbox.pack(side='right')
        high_spinbox.bind('<Return>', lambda e: self.update_step4_params(None, high_var))
        high_spinbox.bind('<FocusOut>', lambda e: self.update_step4_params(None, high_var))

        self.param_widgets = {'low_var': low_var, 'high_var': high_var}

    def create_step5_controls(self):
        """Create controls for Step 5: Watershed"""
        skip_var = tk.BooleanVar(value=not self.params['step_5_watershed']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_5_watershed', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Watershed floods elevation map from markers to segment regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step6_controls(self):
        """Create controls for Step 6: Morphological Close"""
        skip_var = tk.BooleanVar(value=not self.params['step_6_morph_close']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_6_morph_close', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(self.params_frame, text="Number of Iterations:").pack(anchor='w', pady=(5, 2))

        iter_frame = ttk.Frame(self.params_frame)
        iter_frame.pack(fill='x', pady=2)

        iter_var = tk.IntVar(value=self.params['step_6_morph_close']['iterations'])
        iter_scale = ttk.Scale(iter_frame, from_=1, to=10, orient='horizontal',
                              variable=iter_var,
                              command=lambda v: self.update_step6_iterations(iter_var))
        iter_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        iter_spinbox = ttk.Spinbox(iter_frame, from_=1, to=10, textvariable=iter_var,
                                   width=6, command=lambda: self.update_step6_iterations(iter_var))
        iter_spinbox.pack(side='right')
        iter_spinbox.bind('<Return>', lambda e: self.update_step6_iterations(iter_var))
        iter_spinbox.bind('<FocusOut>', lambda e: self.update_step6_iterations(iter_var))

        self.param_widgets = {'iter_var': iter_var}

    def create_step7_controls(self):
        """Create controls for Step 7: Morphological Open"""
        skip_var = tk.BooleanVar(value=not self.params['step_7_morph_open']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_7_morph_open', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(self.params_frame, text="Number of Iterations:").pack(anchor='w', pady=(5, 2))

        iter_frame = ttk.Frame(self.params_frame)
        iter_frame.pack(fill='x', pady=2)

        iter_var = tk.IntVar(value=self.params['step_7_morph_open']['iterations'])
        iter_scale = ttk.Scale(iter_frame, from_=1, to=10, orient='horizontal',
                              variable=iter_var,
                              command=lambda v: self.update_step7_iterations(iter_var))
        iter_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        iter_spinbox = ttk.Spinbox(iter_frame, from_=1, to=10, textvariable=iter_var,
                                   width=6, command=lambda: self.update_step7_iterations(iter_var))
        iter_spinbox.pack(side='right')
        iter_spinbox.bind('<Return>', lambda e: self.update_step7_iterations(iter_var))
        iter_spinbox.bind('<FocusOut>', lambda e: self.update_step7_iterations(iter_var))

        self.param_widgets = {'iter_var': iter_var}

    def create_step8_controls(self):
        """Create controls for Step 8: Fill Holes"""
        skip_var = tk.BooleanVar(value=not self.params['step_8_fillholes']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_8_fillholes', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Fills small holes inside detected regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step9_controls(self):
        """Create controls for Step 9: Labeling"""
        skip_var = tk.BooleanVar(value=not self.params['step_9_labeling']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_9_labeling', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(self.params_frame, text="Minimum Size (pixels):").pack(anchor='w', pady=(5, 2))

        size_frame = ttk.Frame(self.params_frame)
        size_frame.pack(fill='x', pady=2)

        size_var = tk.IntVar(value=self.params['step_9_labeling']['min_size'])
        size_scale = ttk.Scale(size_frame, from_=0, to=200, orient='horizontal',
                              variable=size_var,
                              command=lambda v: self.update_step9_size(size_var))
        size_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        size_spinbox = ttk.Spinbox(size_frame, from_=0, to=200, textvariable=size_var,
                                   width=6, command=lambda: self.update_step9_size(size_var))
        size_spinbox.pack(side='right')
        size_spinbox.bind('<Return>', lambda e: self.update_step9_size(size_var))
        size_spinbox.bind('<FocusOut>', lambda e: self.update_step9_size(size_var))

        self.param_widgets = {'size_var': size_var}

    def create_step10_controls(self):
        """Create controls for Step 10: Final Result"""
        ttk.Label(self.params_frame, text="Minimum Area (pixels):").pack(anchor='w', pady=(5, 2))

        min_area_frame = ttk.Frame(self.params_frame)
        min_area_frame.pack(fill='x', pady=2)

        min_area_var = tk.IntVar(value=self.params['step_10_final']['min_area'])
        min_area_scale = ttk.Scale(min_area_frame, from_=0, to=1500, orient='horizontal',
                              variable=min_area_var,
                              command=lambda v: self.update_step10_min_area(min_area_var))
        min_area_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        min_area_spinbox = ttk.Spinbox(min_area_frame, from_=0, to=1500, textvariable=min_area_var,
                                   width=6, command=lambda: self.update_step10_min_area(min_area_var))
        min_area_spinbox.pack(side='right')
        min_area_spinbox.bind('<Return>', lambda e: self.update_step10_min_area(min_area_var))
        min_area_spinbox.bind('<FocusOut>', lambda e: self.update_step10_min_area(min_area_var))

        ttk.Label(self.params_frame, text="Maximum Area (pixels):").pack(anchor='w', pady=(10, 2))

        max_area_frame = ttk.Frame(self.params_frame)
        max_area_frame.pack(fill='x', pady=2)

        max_area_var = tk.IntVar(value=self.params['step_10_final']['max_area'])
        max_area_scale = ttk.Scale(max_area_frame, from_=1000, to=50000, orient='horizontal',
                              variable=max_area_var,
                              command=lambda v: self.update_step10_max_area(max_area_var))
        max_area_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))

        max_area_spinbox = ttk.Spinbox(max_area_frame, from_=1000, to=50000, textvariable=max_area_var,
                                   width=6, command=lambda: self.update_step10_max_area(max_area_var))
        max_area_spinbox.pack(side='right')
        max_area_spinbox.bind('<Return>', lambda e: self.update_step10_max_area(max_area_var))
        max_area_spinbox.bind('<FocusOut>', lambda e: self.update_step10_max_area(max_area_var))

        ttk.Label(self.params_frame, text="Note: Segments are colored red.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

        self.param_widgets = {'min_area_var': min_area_var, 'max_area_var': max_area_var}

    # Parameter update methods
    def toggle_step_skip(self, step_key, skip_var):
        """Toggle skip for any step"""
        self.params[step_key]['enabled'] = not skip_var.get()
        self.process_all_steps()
        self.display_current_step()

    def update_step4_params(self, low_var, high_var):
        """Update step 4 threshold parameters"""
        try:
            if low_var:
                value = int(low_var.get())
                value = max(0, min(255, value))
                low_var.set(value)
                self.params['step_4_markers']['low_threshold'] = value
            if high_var:
                value = int(high_var.get())
                value = max(0, min(255, value))
                high_var.set(value)
                self.params['step_4_markers']['high_threshold'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            # Reset to current values if invalid input
            if low_var:
                low_var.set(self.params['step_4_markers']['low_threshold'])
            if high_var:
                high_var.set(self.params['step_4_markers']['high_threshold'])

    def update_step6_iterations(self, iter_var):
        """Update step 6 morphological close iterations"""
        try:
            value = int(iter_var.get())
            value = max(1, min(10, value))
            iter_var.set(value)
            self.params['step_6_morph_close']['iterations'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            iter_var.set(self.params['step_6_morph_close']['iterations'])

    def update_step7_iterations(self, iter_var):
        """Update step 7 morphological open iterations"""
        try:
            value = int(iter_var.get())
            value = max(1, min(10, value))
            iter_var.set(value)
            self.params['step_7_morph_open']['iterations'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            iter_var.set(self.params['step_7_morph_open']['iterations'])

    def update_step9_size(self, size_var):
        """Update step 9 minimum size"""
        try:
            value = int(size_var.get())
            value = max(0, min(200, value))
            size_var.set(value)
            self.params['step_9_labeling']['min_size'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            size_var.set(self.params['step_9_labeling']['min_size'])

    def update_step10_min_area(self, area_var):
        """Update step 10 minimum area"""
        try:
            value = int(area_var.get())
            value = max(0, min(1500, value))
            area_var.set(value)
            self.params['step_10_final']['min_area'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            area_var.set(self.params['step_10_final']['min_area'])

    def update_step10_max_area(self, area_var):
        """Update step 10 maximum area"""
        try:
            value = int(area_var.get())
            value = max(1000, min(50000, value))
            area_var.set(value)
            self.params['step_10_final']['max_area'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            area_var.set(self.params['step_10_final']['max_area'])

    def tab2_previous_step(self):
        """Navigate to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.display_current_step()

    def tab2_next_step(self):
        """Navigate to next step"""
        if self.current_step < 10:
            self.current_step += 1
            self.display_current_step()

    def tab2_save_config(self):
        """Save current configuration to JSON"""
        if self.tab2_roi_points is None:
            messagebox.showwarning("Warning", "No ROI coordinates loaded")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if save_path:
            data = {
                "image_path": self.tab2_image_path,
                "polygon_points": self.tab2_roi_points.tolist(),
                "pipeline_params": self.params
            }

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

            messagebox.showinfo("Success", f"Configuration saved to {save_path}")

    def tab2_load_config(self):
        """Load configuration from JSON"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load ROI points
            if 'polygon_points' in data:
                self.tab2_roi_points = np.array(data['polygon_points'], dtype=np.int32)

            # Load parameters
            if 'pipeline_params' in data:
                self.params = data['pipeline_params']

            messagebox.showinfo("Success", "Configuration loaded successfully")
            self.tab2_info_label.config(text="Configuration loaded. Upload image and click Start Processing.")

    # ==================== TAB 3: BATCH PROCESSING ====================

    def init_tab3(self):
        """Initialize Tab 3 - Batch Image Processing with Defect Classification"""
        # Variables
        self.tab3_input_folder = None
        self.tab3_config_data = None
        self.tab3_roi_points = None
        self.tab3_processing = False

        # Main container
        main_container = ttk.Frame(self.tab3)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.LabelFrame(main_container, text="Batch Processing Settings", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))

        # Input folder selection
        folder_frame = ttk.Frame(control_frame)
        folder_frame.pack(fill='x', pady=5)

        ttk.Label(folder_frame, text="Input Folder:").pack(side='left', padx=(0, 10))
        self.tab3_folder_label = ttk.Label(folder_frame, text="No folder selected",
                                            relief='sunken', width=50)
        self.tab3_folder_label.pack(side='left', padx=(0, 10), fill='x', expand=True)
        ttk.Button(folder_frame, text="Select Folder",
                   command=self.tab3_select_folder).pack(side='left')

        # Configuration file selection
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill='x', pady=5)

        ttk.Label(config_frame, text="Configuration:").pack(side='left', padx=(0, 10))
        self.tab3_config_label = ttk.Label(config_frame, text="No configuration loaded",
                                             relief='sunken', width=50)
        self.tab3_config_label.pack(side='left', padx=(0, 10), fill='x', expand=True)
        ttk.Button(config_frame, text="Load Config",
                   command=self.tab3_load_config).pack(side='left')

        # ROI coordinates selection
        roi_frame = ttk.Frame(control_frame)
        roi_frame.pack(fill='x', pady=5)

        ttk.Label(roi_frame, text="ROI Coordinates:").pack(side='left', padx=(0, 10))
        self.tab3_roi_label = ttk.Label(roi_frame, text="No ROI loaded (will use config ROI)",
                                         relief='sunken', width=50)
        self.tab3_roi_label.pack(side='left', padx=(0, 10), fill='x', expand=True)
        ttk.Button(roi_frame, text="Load ROI",
                   command=self.tab3_load_roi).pack(side='left')

        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', pady=10)

        self.tab3_process_button = ttk.Button(button_frame, text="Start Batch Processing",
                                               command=self.tab3_start_batch_processing,
                                               state='disabled')
        self.tab3_process_button.pack(side='left', padx=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(main_container, text="Processing Status", padding=10)
        progress_frame.pack(fill='both', expand=True)

        # Progress bar
        self.tab3_progress = ttk.Progressbar(progress_frame, orient='horizontal',
                                              length=300, mode='determinate')
        self.tab3_progress.pack(fill='x', pady=5)

        # Status label
        self.tab3_status_label = ttk.Label(progress_frame, text="Ready",
                                            font=('Arial', 10, 'bold'))
        self.tab3_status_label.pack(pady=5)

        # Log text area
        log_frame = ttk.Frame(progress_frame)
        log_frame.pack(fill='both', expand=True, pady=5)

        log_scrollbar = ttk.Scrollbar(log_frame)
        log_scrollbar.pack(side='right', fill='y')

        self.tab3_log_text = scrolledtext.ScrolledText(log_frame, height=20,
                                                         yscrollcommand=log_scrollbar.set)
        self.tab3_log_text.pack(fill='both', expand=True)
        log_scrollbar.config(command=self.tab3_log_text.yview)

    def tab3_select_folder(self):
        """Select input folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Input Folder")

        if folder_path:
            self.tab3_input_folder = folder_path
            self.tab3_folder_label.config(text=folder_path)
            self.tab3_log(f"Input folder selected: {folder_path}")
            self.tab3_check_ready()

    def tab3_load_config(self):
        """Load configuration file for batch processing"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.tab3_config_data = json.load(f)

                # Verify it has required keys
                if 'polygon_points' not in self.tab3_config_data or 'pipeline_params' not in self.tab3_config_data:
                    messagebox.showerror("Error",
                                          "Invalid configuration file. Must contain polygon_points and pipeline_params.")
                    self.tab3_config_data = None
                    return

                self.tab3_config_label.config(text=file_path)
                self.tab3_log(f"Configuration loaded: {file_path}")
                self.tab3_check_ready()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
                self.tab3_config_data = None

    def tab3_load_roi(self):
        """Load ROI coordinates from JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select ROI Coordinates File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Can load from either pure ROI file or config file
                if 'polygon_points' in data:
                    self.tab3_roi_points = np.array(data['polygon_points'], dtype=np.int32)
                    self.tab3_roi_label.config(text=file_path)
                    self.tab3_log(f"ROI coordinates loaded: {file_path} ({len(self.tab3_roi_points)} points)")
                    self.tab3_check_ready()
                else:
                    messagebox.showerror("Error", "Invalid file. Must contain 'polygon_points'.")
                    self.tab3_roi_points = None
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROI: {str(e)}")
                self.tab3_roi_points = None

    def tab3_check_ready(self):
        """Check if ready to start batch processing"""
        # Need: input folder + config (for parameters)
        # ROI is optional - will use separate ROI if loaded, otherwise config ROI
        if self.tab3_input_folder and self.tab3_config_data:
            # Check if we have ROI (either separate or in config)
            has_roi = self.tab3_roi_points is not None or 'polygon_points' in self.tab3_config_data
            if has_roi:
                self.tab3_process_button.config(state='normal')
            else:
                self.tab3_process_button.config(state='disabled')
        else:
            self.tab3_process_button.config(state='disabled')

    def tab3_log(self, message):
        """Add message to log"""
        self.tab3_log_text.insert(tk.END, message + '\n')
        self.tab3_log_text.see(tk.END)
        self.tab3_log_text.update()

    def tab3_start_batch_processing(self):
        """Start batch processing of images"""
        if self.tab3_processing:
            return

        self.tab3_processing = True
        self.tab3_process_button.config(state='disabled')

        # Get list of image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for root, dirs, files in os.walk(self.tab3_input_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)

        if not image_files:
            messagebox.showwarning("Warning", "No image files found in selected folder")
            self.tab3_processing = False
            self.tab3_process_button.config(state='normal')
            return

        self.tab3_log(f"\nFound {len(image_files)} image(s) to process")

        # Create results folder structure
        results_path = os.path.join(self.tab3_input_folder, "results")
        defects_path = os.path.join(results_path, "defects")
        non_defects_path = os.path.join(results_path, "non_defects")

        os.makedirs(defects_path, exist_ok=True)
        os.makedirs(non_defects_path, exist_ok=True)

        self.tab3_log(f"Created output folders:")
        self.tab3_log(f"  - {defects_path}")
        self.tab3_log(f"  - {non_defects_path}")

        # Process each image
        total = len(image_files)
        defect_count = 0
        non_defect_count = 0

        for idx, filename in enumerate(image_files):
            # Update progress
            progress = (idx / total) * 100
            self.tab3_progress['value'] = progress
            self.tab3_status_label.config(text=f"Processing {idx + 1}/{total}: {filename}")
            self.root.update()

            # Process image
            input_path = os.path.join(self.tab3_input_folder, filename)
            try:
                num_segments = self.tab3_process_single_image(input_path,
                                                                defects_path,
                                                                non_defects_path,
                                                                filename)

                if num_segments >= 1:
                    defect_count += 1
                    self.tab3_log(f"✓ {filename} -> defects ({num_segments} segments)")
                else:
                    non_defect_count += 1
                    self.tab3_log(f"✓ {filename} -> non_defects ({num_segments} segment(s))")

            except Exception as e:
                self.tab3_log(f"✗ {filename} -> ERROR: {str(e)}")

        # Complete
        self.tab3_progress['value'] = 100
        self.tab3_status_label.config(text="Processing Complete!")
        self.tab3_log(f"\n=== Processing Complete ===")
        self.tab3_log(f"Total processed: {total}")
        self.tab3_log(f"Defects: {defect_count}")
        self.tab3_log(f"Non-defects: {non_defect_count}")

        self.tab3_processing = False
        self.tab3_process_button.config(state='normal')

        messagebox.showinfo("Complete",
                             f"Batch processing complete!\n\n"
                             f"Processed: {total} images\n"
                             f"Defects: {defect_count}\n"
                             f"Non-defects: {non_defect_count}")

    def tab3_process_single_image(self, image_path, defects_folder, non_defects_folder, filename):
        """Process a single image and save to appropriate folder"""
        # Load image
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise ValueError("Failed to load image")

        # Get ROI points - use separate ROI if loaded, otherwise use config ROI
        if self.tab3_roi_points is not None:
            roi_points = self.tab3_roi_points
        else:
            roi_points = np.array(self.tab3_config_data['polygon_points'], dtype=np.int32)

        # Get parameters from config
        params = self.tab3_config_data['pipeline_params']

        # Apply mask
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        masked_image = cv_image.copy()
        masked_image[mask == 0] = [0, 0, 0]

        # Process through watershed pipeline (simplified version for batch)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Sobel gradient
        if params['step_3_sobel']['enabled']:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            elevation_map = np.hypot(sobel_x, sobel_y)
            elevation_display = cv2.normalize(elevation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            elevation_display = gray

        # Histogram markers with varying values
        if params['step_4_markers']['enabled']:
            low_thresh = params['step_4_markers']['low_threshold']
            high_thresh = params['step_4_markers']['high_threshold']
            
            inv = 255 - gray
            background = cv2.GaussianBlur(inv, (0,0), sigmaX=50, sigmaY=50)
            gbImg = cv2.subtract(inv, background)
            _, gb = cv2.threshold(gbImg, 10, 255, cv2.THRESH_BINARY)
            
            
            markers = np.zeros_like(gray, dtype=np.int32)
            background_mask = gray < low_thresh
            markers[background_mask] = gb[background_mask]
            
            foreground_mask = gray > high_thresh
            markers[foreground_mask] = gray[foreground_mask].astype(np.int32) + 1
        else:
            markers = np.zeros_like(gray, dtype=np.int32)
            markers[gray > 0] = gray[gray > 0].astype(np.int32) + 1

        # Watershed
        if params['step_5_watershed']['enabled']:
            elevation_map_color = cv2.cvtColor(elevation_display, cv2.COLOR_GRAY2BGR)
            markers_watershed = markers.copy()
            segmentation = cv2.watershed(elevation_map_color, markers_watershed)
        else:
            segmentation = markers.copy()

        # Convert segmentation to binary
        # Foreground regions have marker values corresponding to high intensities (> high_threshold)
        if params['step_4_markers']['enabled']:
            high_thresh = params['step_4_markers']['high_threshold']
            binary_seg = (segmentation > high_thresh).astype(np.uint8)
        else:
            binary_seg = (segmentation > 0).astype(np.uint8)

        # Morphological Close
        if params['step_6_morph_close']['enabled']:
            iterations = params['step_6_morph_close']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(binary_seg, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            morph_close = binary_seg.copy()

        # Morphological Open
        if params['step_7_morph_open']['enabled']:
            iterations = params['step_7_morph_open']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            morph_open = morph_close.copy()

        # Fill holes
        if params['step_8_fillholes']['enabled']:
            filled = ndi.binary_fill_holes(morph_open).astype(np.uint8) * 255
        else:
            filled = morph_open * 255

        # Label components
        if params['step_9_labeling']['enabled']:
            labeled_array, num_features = ndi.label(filled)
            min_size = params['step_9_labeling']['min_size']
            sizes = np.bincount(labeled_array.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0
            cleaned_labels = mask_sizes[labeled_array]
            labeled_cleaned, num_cleaned = ndi.label(cleaned_labels)
        else:
            labeled_cleaned, num_cleaned = ndi.label(filled)

        # Color segments
        #result = masked_image.copy()
        result = cv_image.copy()
        segment_color = (0, 0, 255)  # Red in BGR
        min_area = params['step_10_final']['min_area']
        max_area = params['step_10_final']['max_area']

        valid_segments = 0
        for region_label in range(1, num_cleaned + 1):
            region_mask = (labeled_cleaned == region_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area <= area <= max_area:
                        valid_segments += 1
                        # Color the segment
                        cv2.drawContours(result, [contour], -1, segment_color, -1)

        # Save to appropriate folder
        filename = os.path.basename(filename)
        if valid_segments >= 1:
            output_path = os.path.join(defects_folder, filename)
        else:
            output_path = os.path.join(non_defects_folder, filename)

        cv2.imwrite(output_path, result)

        return valid_segments


def main():
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
