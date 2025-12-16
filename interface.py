import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import os
import subprocess
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
from scipy.spatial import cKDTree
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')


class CloudCompareInterface:
    """CloudCompare interface class for visualizing clustering results"""

    def __init__(self, cloudcompare_path=None):
        """
        Initialize CloudCompare interface
        Args:
            cloudcompare_path (str): CloudCompare executable path
        """
        self.cloudcompare_path = cloudcompare_path or self._find_cloudcompare()
        self.temp_dir = Path("temp_clustering")
        self.temp_dir.mkdir(exist_ok=True)
        self.process = None

    def _find_cloudcompare(self):
        """Automatically find CloudCompare installation path"""
        possible_paths = [
            "E:/CloudCompare/CloudCompare.exe",
            "C:/Program Files/CloudCompare/CloudCompare.exe",
            "C:/Program Files (x86)/CloudCompare/CloudCompare.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "CloudCompare"  # Assume it's in PATH

    def save_colored_pointcloud(self, points, normals, labels, filename):
        """
        Save colored point cloud file (with normal information)
        Args:
            points (np.array): Point cloud coordinates (N x 3)
            normals (np.array): Normal vectors (N x 3)
            labels (np.array): Cluster labels
            filename (str): Output filename
        """
        # Assign colors to different clusters
        unique_labels = np.unique(labels)
        colors = self._generate_colors(len(unique_labels))

        # Create colored point cloud data
        colored_data = []
        for i, (point, normal, label) in enumerate(zip(points, normals, labels)):
            if label == -1:  # Noise points set to gray
                color = [128, 128, 128]
            else:
                color_idx = np.where(unique_labels == label)[0][0]
                color = colors[color_idx % len(colors)]

            # Format: x y z nx ny nz r g b
            colored_data.append([
                point[0], point[1], point[2],
                normal[0], normal[1], normal[2],
                color[0], color[1], color[2]
            ])

        # Save as PLY format
        output_path = self.temp_dir / filename
        self._save_ply_with_normals_and_colors(colored_data, output_path)
        return str(output_path)

    def _generate_colors(self, n_colors):
        """Generate n different colors using golden angle distribution"""
        colors = []
        for i in range(n_colors):
            hue = (i * 137.508) % 360  # Golden angle distribution
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append([int(c * 255) for c in rgb])
        return colors

    def _hsv_to_rgb(self, h, s, v):
        """HSV to RGB color space conversion"""
        import colorsys
        return colorsys.hsv_to_rgb(h / 360, s, v)

    def _save_ply_with_normals_and_colors(self, data, filepath):
        """Save PLY format file with normals and colors"""
        data = np.array(data)
        n_points = len(data)

        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write point cloud data
            for row in data:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
                        f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f} "
                        f"{int(row[6])} {int(row[7])} {int(row[8])}\n")

    def open_in_cloudcompare(self, filepath):
        """Open file in CloudCompare"""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                time.sleep(0.5)

            cmd = [self.cloudcompare_path, filepath]
            self.process = subprocess.Popen(cmd)
            return True
        except Exception as e:
            return False


class OptimizedDBSCAN:
    """Optimized DBSCAN implementation for large-scale point clouds"""

    def __init__(self, eps=0.1, min_samples=10, max_points_per_batch=50000,
                 use_voxel_downsampling=True, voxel_size=None):
        """
        Initialize optimized DBSCAN
        Args:
            eps: Neighborhood radius
            min_samples: Minimum samples
            max_points_per_batch: Maximum points per batch
            use_voxel_downsampling: Whether to use voxel downsampling
            voxel_size: Voxel size (None for auto calculation)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_points_per_batch = max_points_per_batch
        self.use_voxel_downsampling = use_voxel_downsampling
        self.voxel_size = voxel_size

    def voxel_downsample(self, points, voxel_size=None):
        """
        Voxel downsampling
        Args:
            points: Point cloud data
            voxel_size: Voxel size
        Returns:
            downsampled_points: Downsampled points
            point_to_voxel: Mapping from original points to voxels
        """
        if voxel_size is None:
            # Auto calculate voxel size (half of eps)
            voxel_size = self.eps * 0.5

        # Calculate voxel indices for each point
        min_bound = np.min(points, axis=0)
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)

        # Use dictionary to store points in each voxel
        voxel_dict = {}
        point_to_voxel = np.zeros(len(points), dtype=np.int32)

        for i, voxel_idx in enumerate(voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(i)

        # Calculate center point for each voxel
        downsampled_points = []
        voxel_to_points = {}
        voxel_id = 0

        for voxel_key, point_indices in voxel_dict.items():
            voxel_center = np.mean(points[point_indices], axis=0)
            downsampled_points.append(voxel_center)
            voxel_to_points[voxel_id] = point_indices
            for idx in point_indices:
                point_to_voxel[idx] = voxel_id
            voxel_id += 1

        return np.array(downsampled_points), point_to_voxel, voxel_to_points

    def fit_predict(self, points, callback=None):
        """
        Execute DBSCAN clustering
        Args:
            points: Point cloud data
            callback: Progress callback function
        Returns:
            labels: Cluster labels
        """
        n_points = len(points)

        # For small point clouds, use standard DBSCAN directly
        if n_points <= self.max_points_per_batch:
            if callback:
                callback(0.3, "Executing standard DBSCAN...")
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
            labels = clustering.fit_predict(points)
            if callback:
                callback(1.0, "Clustering completed")
            return labels

        # For large-scale point clouds, use optimization strategy
        if self.use_voxel_downsampling:
            if callback:
                callback(0.1, f"Downsampling {n_points} points...")

            # Voxel downsampling
            downsampled_points, point_to_voxel, voxel_to_points = self.voxel_downsample(
                points, self.voxel_size
            )

            if callback:
                callback(0.3, f"Downsampled to {len(downsampled_points)} voxel centers")

            # Cluster downsampled points
            clustering = DBSCAN(eps=self.eps * 1.5, min_samples=max(3, self.min_samples // 2), n_jobs=-1)
            voxel_labels = clustering.fit_predict(downsampled_points)

            if callback:
                callback(0.6, "Mapping labels to original points...")

            # Map labels back to original points
            labels = np.zeros(n_points, dtype=np.int32)
            for i in range(n_points):
                voxel_id = point_to_voxel[i]
                labels[i] = voxel_labels[voxel_id]

            # Post-processing: refine boundaries
            if callback:
                callback(0.8, "Refining cluster boundaries...")
            labels = self._refine_boundaries(points, labels)

        else:
            # Batch processing strategy
            labels = self._batch_dbscan(points, callback)

        if callback:
            callback(1.0, "Clustering completed")
        return labels

    def _batch_dbscan(self, points, callback=None):
        """
        Execute DBSCAN in batches
        """
        n_points = len(points)
        labels = np.full(n_points, -1, dtype=np.int32)
        current_label = 0

        # Build KD tree for accelerated neighborhood search
        if callback:
            callback(0.1, "Building spatial index...")
        kdtree = cKDTree(points)

        # Randomly shuffle point order
        indices = np.random.permutation(n_points)
        batch_size = self.max_points_per_batch
        n_batches = (n_points + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            if callback:
                progress = 0.1 + 0.8 * batch_idx / n_batches
                callback(progress, f"Processing batch {batch_idx + 1}/{n_batches}")

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_points)
            batch_indices = indices[start_idx:end_idx]

            # Get batch points and their neighborhoods
            batch_points = points[batch_indices]

            # Extend batch to include neighborhood points
            extended_indices = set(batch_indices)
            for idx in batch_indices:
                neighbors = kdtree.query_ball_point(points[idx], self.eps)
                extended_indices.update(neighbors)

            extended_indices = list(extended_indices)
            extended_points = points[extended_indices]

            # Cluster extended batch
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
            batch_labels = clustering.fit_predict(extended_points)

            # Map labels
            for i, idx in enumerate(extended_indices):
                if labels[idx] == -1 and batch_labels[i] != -1:
                    labels[idx] = current_label + batch_labels[i]

            # Update label count
            if len(batch_labels[batch_labels != -1]) > 0:
                current_label += max(batch_labels) + 1

        return labels

    def _refine_boundaries(self, points, labels, iterations=1):
        """
        Refine cluster boundaries
        """
        refined_labels = labels.copy()

        for _ in range(iterations):
            # Build KD tree
            kdtree = cKDTree(points)

            # For each noise point, check if it should be assigned to nearby clusters
            noise_mask = refined_labels == -1
            noise_indices = np.where(noise_mask)[0]

            for idx in noise_indices:
                # Find points in neighborhood
                neighbors = kdtree.query_ball_point(points[idx], self.eps)
                neighbor_labels = refined_labels[neighbors]

                # Count non-noise neighbor labels
                valid_labels = neighbor_labels[neighbor_labels != -1]
                if len(valid_labels) >= self.min_samples:
                    # Assign to most common label
                    unique, counts = np.unique(valid_labels, return_counts=True)
                    refined_labels[idx] = unique[np.argmax(counts)]

        return refined_labels


class RealtimeDBSCANTool:
    """DBSCAN real-time clustering parameter tuning tool main class"""

    def __init__(self):
        self.cc_interface = CloudCompareInterface()
        self.original_data = None  # Original data (N x 10: x y z r g b label nx ny nz)
        self.current_category = 'leaf'  # Current processing category
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("DBSCAN Clustering Tool - Large-scale Point Cloud Optimized")
        self.root.geometry("1400x800")

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left and right panels using PanedWindow
        paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)

        # Right panel for preview
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)

        # Setup left panel controls
        self._setup_control_panels(left_frame)

        # Setup right panel preview
        self._setup_preview_panel(right_frame)

        # Set initial pane sizes
        self.root.after(100, lambda: paned_window.sashpos(0, 700))

    def _setup_control_panels(self, parent):
        """Setup control panels in left frame"""
        # File selection area
        self._create_file_selection_frame(parent)

        # Category selection area
        self._create_category_selection_frame(parent)

        # DBSCAN parameters area
        self._create_dbscan_parameters_frame(parent)

        # Optimization options area
        self._create_optimization_frame(parent)

        # Control buttons area
        self._create_control_buttons_frame(parent)

        # Progress bar
        self._create_progress_bar(parent)

        # Status bar and results display
        self._create_status_and_results_frame(parent)

    def _setup_preview_panel(self, parent):
        """Setup preview panel in right frame"""
        # Preview title
        preview_title = ttk.LabelFrame(parent, text="Point Cloud Preview", padding=10)
        preview_title.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.preview_figure = Figure(figsize=(8, 6), dpi=80, facecolor='white')
        self.preview_ax = self.preview_figure.add_subplot(111, projection='3d')

        # Create canvas
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, preview_title)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Preview control buttons
        preview_control_frame = ttk.Frame(preview_title)
        preview_control_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(preview_control_frame, text="Refresh Preview",
                   command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preview_control_frame, text="Reset View",
                   command=self.reset_preview_view).pack(side=tk.LEFT, padx=(0, 5))

        # Sample size control
        ttk.Label(preview_control_frame, text="Sample Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.preview_sample_var = tk.IntVar(value=5000)
        sample_combo = ttk.Combobox(preview_control_frame, textvariable=self.preview_sample_var,
                                    values=[1000, 2000, 5000, 10000, 20000], width=8, state='readonly')
        sample_combo.pack(side=tk.LEFT)
        sample_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_preview())

        # Initialize empty preview
        self._show_empty_preview()

    def _create_file_selection_frame(self, parent):
        """Create file selection area"""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Select Point Cloud File", command=self.load_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))

    def _create_category_selection_frame(self, parent):
        """Create category selection area"""
        category_frame = ttk.LabelFrame(parent, text="Category Selection", padding=10)
        category_frame.pack(fill=tk.X, pady=(0, 10))

        self.category_var = tk.StringVar(value='leaf')
        ttk.Radiobutton(category_frame, text="Leaf (Label 0)", variable=self.category_var,
                        value='leaf', command=self.on_category_change).pack(side=tk.LEFT)
        ttk.Radiobutton(category_frame, text="Petiole (Label 1)", variable=self.category_var,
                        value='petiole', command=self.on_category_change).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(category_frame, text="Trunk (Label 2)", variable=self.category_var,
                        value='trunk', command=self.on_category_change).pack(side=tk.LEFT, padx=(20, 0))

    def _create_dbscan_parameters_frame(self, parent):
        """Create DBSCAN parameters area"""
        params_frame = ttk.LabelFrame(parent, text="DBSCAN Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        self.param_vars = {}

        # eps parameter - neighborhood radius
        row = 0
        ttk.Label(params_frame, text="eps (Neighborhood Radius):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.param_vars['eps'] = tk.DoubleVar(value=0.1)
        eps_scale = ttk.Scale(params_frame, from_=0.01, to=0.5,
                              variable=self.param_vars['eps'],
                              orient=tk.HORIZONTAL, length=300)
        eps_scale.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        eps_label = ttk.Label(params_frame, textvariable=self.param_vars['eps'])
        eps_label.grid(row=row, column=2, sticky=tk.W, padx=(10, 0))

        # min_samples parameter - minimum samples
        row = 1
        ttk.Label(params_frame, text="min_samples (Minimum Samples):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.param_vars['min_samples'] = tk.IntVar(value=10)
        samples_scale = ttk.Scale(params_frame, from_=3, to=50,
                                  variable=self.param_vars['min_samples'],
                                  orient=tk.HORIZONTAL, length=300)
        samples_scale.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        samples_label = ttk.Label(params_frame, textvariable=self.param_vars['min_samples'])
        samples_label.grid(row=row, column=2, sticky=tk.W, padx=(10, 0))

        # Quick parameter setting buttons
        row = 2
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Fine Clustering",
                   command=lambda: self.set_quick_params(0.05, 15)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Standard Clustering",
                   command=lambda: self.set_quick_params(0.1, 10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Coarse Clustering",
                   command=lambda: self.set_quick_params(0.2, 5)).pack(side=tk.LEFT, padx=5)

    def _create_optimization_frame(self, parent):
        """Create optimization options area"""
        opt_frame = ttk.LabelFrame(parent, text="Large-scale Point Cloud Optimization Options", padding=10)
        opt_frame.pack(fill=tk.X, pady=(0, 10))

        # Enable optimization
        self.use_optimization = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Enable large-scale point cloud optimization",
                        variable=self.use_optimization,
                        command=self.on_optimization_change).grid(row=0, column=0, sticky=tk.W)

        # Optimization method selection
        ttk.Label(opt_frame, text="Optimization Method:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.optimization_method = tk.StringVar(value='voxel')
        ttk.Radiobutton(opt_frame, text="Voxel Downsampling",
                        variable=self.optimization_method,
                        value='voxel').grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(opt_frame, text="Batch Processing",
                        variable=self.optimization_method,
                        value='batch').grid(row=1, column=2, sticky=tk.W)

        # Voxel size
        ttk.Label(opt_frame, text="Voxel Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.voxel_size_var = tk.DoubleVar(value=0.05)
        voxel_scale = ttk.Scale(opt_frame, from_=0.01, to=0.2,
                                variable=self.voxel_size_var,
                                orient=tk.HORIZONTAL, length=200)
        voxel_scale.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(opt_frame, textvariable=self.voxel_size_var).grid(row=2, column=2, sticky=tk.W, padx=(10, 0))

        # Batch processing size
        ttk.Label(opt_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=50000)
        batch_combo = ttk.Combobox(opt_frame, textvariable=self.batch_size_var,
                                   values=[10000, 20000, 30000, 50000, 80000, 100000],
                                   width=15, state='readonly')
        batch_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))

        # Point cloud information display
        self.points_info_label = ttk.Label(opt_frame, text="Point Cloud Info: Not loaded", foreground="blue")
        self.points_info_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=10)

    def _create_control_buttons_frame(self, parent):
        """Create control buttons area"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        buttons = [
            ("Apply Clustering", self.apply_clustering),
            ("View in CloudCompare", self.open_in_cc),
            ("Save Cluster Files", self.save_result),
            ("Save Configuration", self.save_config),
            ("Load Configuration", self.load_config)
        ]

        for text, command in buttons:
            ttk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT, padx=(0, 10))

    def _create_progress_bar(self, parent):
        """Create progress bar"""
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=400)
        self.progress_bar.pack(side=tk.LEFT)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side=tk.LEFT, padx=(10, 0))

    def _create_status_and_results_frame(self, parent):
        """Create status bar and results display area"""
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 5))

        # Results display area (smaller height due to preview panel)
        result_frame = ttk.LabelFrame(parent, text="Results", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, height=6, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _show_empty_preview(self):
        """Show empty preview with instructions"""
        self.preview_ax.clear()
        self.preview_ax.text(0.5, 0.5, 0.5, 'Load a point cloud file\nto see preview',
                             transform=self.preview_ax.transAxes, fontsize=12, ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        self.preview_ax.set_xlabel('X')
        self.preview_ax.set_ylabel('Y')
        self.preview_ax.set_zlabel('Z')
        self.preview_ax.set_title('Point Cloud Preview')
        self.preview_canvas.draw()

    def refresh_preview(self):
        """Refresh point cloud preview"""
        if self.original_data is None:
            self._show_empty_preview()
            return

        current_data = self.get_current_data()
        if current_data is None or len(current_data) == 0:
            self.preview_ax.clear()
            self.preview_ax.text(0.5, 0.5, 0.5, f'No data for category:\n{self.current_category}',
                                 transform=self.preview_ax.transAxes, fontsize=12, ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            self.preview_ax.set_title('Point Cloud Preview')
            self.preview_canvas.draw()
            return

        # Sample data for display
        sample_size = min(self.preview_sample_var.get(), len(current_data))
        if len(current_data) > sample_size:
            indices = np.random.choice(len(current_data), sample_size, replace=False)
            display_data = current_data[indices]
        else:
            display_data = current_data

        xyz = display_data[:, :3]
        colors = display_data[:, 3:6] / 255.0  # Normalize RGB values

        self.preview_ax.clear()
        self.preview_ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                c=colors, s=1, alpha=0.6)

        self.preview_ax.set_xlabel('X')
        self.preview_ax.set_ylabel('Y')
        self.preview_ax.set_zlabel('Z')
        self.preview_ax.set_title(
            f'{self.current_category.title()} Point Cloud\n({len(display_data):,} / {len(current_data):,} points)')

        # Set equal aspect ratio
        self._set_equal_aspect_3d()
        self.preview_canvas.draw()

    def refresh_clustering_preview(self, xyz_data, labels):
        """Refresh preview with clustering results"""
        if xyz_data is None or labels is None:
            return

        # Sample data for display
        sample_size = min(self.preview_sample_var.get(), len(xyz_data))
        if len(xyz_data) > sample_size:
            indices = np.random.choice(len(xyz_data), sample_size, replace=False)
            display_xyz = xyz_data[indices]
            display_labels = labels[indices]
        else:
            display_xyz = xyz_data
            display_labels = labels

        self.preview_ax.clear()

        # Generate colors for clusters
        unique_labels = np.unique(display_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 0:
            # Use colormap for clusters
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Noise points in gray
                    cluster_mask = display_labels == label
                    if np.any(cluster_mask):
                        self.preview_ax.scatter(display_xyz[cluster_mask, 0],
                                                display_xyz[cluster_mask, 1],
                                                display_xyz[cluster_mask, 2],
                                                c='gray', s=1, alpha=0.3, label='Noise')
                else:
                    cluster_mask = display_labels == label
                    if np.any(cluster_mask):
                        color_idx = np.where(unique_labels[unique_labels != -1] == label)[0][0]
                        self.preview_ax.scatter(display_xyz[cluster_mask, 0],
                                                display_xyz[cluster_mask, 1],
                                                display_xyz[cluster_mask, 2],
                                                c=[colors[color_idx % len(colors)]], s=2, alpha=0.7,
                                                label=f'Cluster {label}' if i < 10 else None)  # Limit legend entries

        self.preview_ax.set_xlabel('X')
        self.preview_ax.set_ylabel('Y')
        self.preview_ax.set_zlabel('Z')
        self.preview_ax.set_title(
            f'Clustering Results\n{n_clusters} clusters, {np.sum(display_labels == -1)} noise points')

        # Add legend if not too many clusters
        if n_clusters <= 10:
            self.preview_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        self._set_equal_aspect_3d()
        self.preview_canvas.draw()

    def _set_equal_aspect_3d(self):
        """Set equal aspect ratio for 3D plot"""
        try:
            # Get current axis limits
            xlims = self.preview_ax.get_xlim3d()
            ylims = self.preview_ax.get_ylim3d()
            zlims = self.preview_ax.get_zlim3d()

            # Calculate ranges
            xrange = xlims[1] - xlims[0]
            yrange = ylims[1] - ylims[0]
            zrange = zlims[1] - zlims[0]

            # Get the maximum range
            max_range = max(xrange, yrange, zrange)

            # Calculate centers
            xcenter = (xlims[1] + xlims[0]) / 2
            ycenter = (ylims[1] + ylims[0]) / 2
            zcenter = (zlims[1] + zlims[0]) / 2

            # Set equal limits
            self.preview_ax.set_xlim3d(xcenter - max_range / 2, xcenter + max_range / 2)
            self.preview_ax.set_ylim3d(ycenter - max_range / 2, ycenter + max_range / 2)
            self.preview_ax.set_zlim3d(zcenter - max_range / 2, zcenter + max_range / 2)
        except:
            pass  # If setting equal aspect fails, just continue

    def reset_preview_view(self):
        """Reset preview view to default"""
        if hasattr(self, 'preview_ax'):
            self.preview_ax.view_init(elev=20, azim=45)
            self.preview_canvas.draw()

    def set_quick_params(self, eps, min_samples):
        """Quick parameter setting"""
        self.param_vars['eps'].set(eps)
        self.param_vars['min_samples'].set(min_samples)

    def on_category_change(self):
        """Callback when category changes"""
        self.current_category = self.category_var.get()
        self.update_data_info()
        self.refresh_preview()  # Refresh preview when category changes

    def on_optimization_change(self):
        """Callback when optimization options change"""
        # Update interface state
        pass

    def update_progress(self, value, text=""):
        """Update progress bar"""
        self.progress_var.set(value * 100)
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    def load_file(self):
        """Load point cloud file (supports 10-column format: x y z r g b label nx ny nz)"""
        filename = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Try to load data
                self.status_var.set("Loading file...")
                self.root.update()

                data = np.loadtxt(filename)
                if data.shape[1] != 10:
                    messagebox.showerror("Error",
                                         "File format error, requires 10 columns (x, y, z, r, g, b, label, nx, ny, nz)")
                    return

                self.original_data = data
                self.file_label.config(text=f"Loaded: {Path(filename).name}")
                self.status_var.set(f"Loaded {len(data)} points")

                # Update data information
                self.update_data_info()

                # Show data statistics
                self.show_data_statistics()

                # Refresh preview
                self.refresh_preview()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                self.status_var.set("Loading failed")

    def update_data_info(self):
        """Update current data information display"""
        if self.original_data is None:
            return

        current_data = self.get_current_data()
        if current_data is not None and len(current_data) > 0:
            n_points = len(current_data)
            info_text = f"Current category points: {n_points:,}"

            if n_points > 50000:
                info_text += " (optimization recommended)"
                self.points_info_label.config(text=info_text, foreground="red")
            elif n_points > 20000:
                info_text += " (may need optimization)"
                self.points_info_label.config(text=info_text, foreground="orange")
            else:
                info_text += " (can process directly)"
                self.points_info_label.config(text=info_text, foreground="green")

    def show_data_statistics(self):
        """Show data statistics"""
        if self.original_data is None:
            return

        stats_text = "Data Statistics:\n"
        stats_text += "=" * 50 + "\n"

        labels = [0, 1, 2]
        names = ['Leaf', 'Petiole', 'Trunk']

        for label, name in zip(labels, names):
            count = np.sum(self.original_data[:, 6] == label)  # Label in 7th column (index 6)
            percentage = count / len(self.original_data) * 100
            stats_text += f"{name} (Label {label}): {count:,} points ({percentage:.1f}%)\n"

        # Add normal vector statistics
        normals = self.original_data[:, 7:10]  # Normals in columns 8-10 (index 7-9)
        normal_magnitude = np.linalg.norm(normals, axis=1)
        stats_text += f"\nNormal Vector Statistics:\n"
        stats_text += f"Average length: {np.mean(normal_magnitude):.3f}\n"
        stats_text += f"Length range: [{np.min(normal_magnitude):.3f}, {np.max(normal_magnitude):.3f}]\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, stats_text)

    def get_current_data(self):
        """Get data for currently selected category"""
        if self.original_data is None:
            return None

        label_map = {'leaf': 0, 'petiole': 1, 'trunk': 2}
        target_label = label_map.get(self.current_category, 0)

        mask = self.original_data[:, 6] == target_label  # Label in 7th column (index 6)
        return self.original_data[mask]

    def apply_clustering(self):
        """Apply DBSCAN clustering algorithm"""
        current_data = self.get_current_data()
        if current_data is None:
            messagebox.showwarning("Warning", "Please load point cloud file first")
            return

        if len(current_data) == 0:
            messagebox.showwarning("Warning", f"Current category ({self.current_category}) has no data")
            return

        # Reset progress bar
        self.progress_var.set(0)
        self.progress_label.config(text="")

        self.status_var.set("Clustering...")
        self.root.update()

        # Execute clustering in separate thread to avoid interface freezing
        threading.Thread(target=self._perform_clustering, args=(current_data,), daemon=True).start()

    def _perform_clustering(self, data):
        """Execute clustering in separate thread"""
        try:
            xyz_data = data[:, :3]  # Extract coordinates (x, y, z)
            normals_data = data[:, 7:10]  # Extract normals (nx, ny, nz)

            n_points = len(xyz_data)

            # Get parameters
            eps = self.param_vars['eps'].get()
            min_samples = self.param_vars['min_samples'].get()

            # Progress callback function
            def progress_callback(progress, message):
                self.root.after(0, lambda: self.update_progress(progress, message))

            # Choose clustering strategy based on point count and optimization settings
            if self.use_optimization.get() and n_points > 20000:
                # Use optimized DBSCAN
                progress_callback(0.05, f"Optimized processing {n_points:,} points...")

                use_voxel = self.optimization_method.get() == 'voxel'
                voxel_size = self.voxel_size_var.get() if use_voxel else None
                batch_size = self.batch_size_var.get()

                optimized_dbscan = OptimizedDBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    max_points_per_batch=batch_size,
                    use_voxel_downsampling=use_voxel,
                    voxel_size=voxel_size
                )

                labels = optimized_dbscan.fit_predict(xyz_data, progress_callback)

            else:
                # Use standard DBSCAN
                progress_callback(0.1, f"Standard DBSCAN processing {n_points:,} points...")

                clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = clustering.fit_predict(xyz_data)

                progress_callback(1.0, "Clustering completed")

            # Save result file
            output_file = f"{self.current_category}_dbscan_result.ply"
            filepath = self.cc_interface.save_colored_pointcloud(xyz_data, normals_data, labels, output_file)

            # Update UI (in main thread)
            self.root.after(0, self._clustering_completed, labels, filepath)

        except Exception as e:
            self.root.after(0, lambda: self._clustering_error(str(e)))

    def _clustering_completed(self, labels, filepath):
        """Clustering completion callback"""
        # Save clustering result data for later use
        current_data = self.get_current_data()
        if current_data is not None:
            self.current_result_labels = labels
            self.current_result_data = current_data

        # Statistics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)

        result_text = f"\n{'=' * 50}\n"
        result_text += f"Clustering Results:\n"
        result_text += f"Number of clusters: {n_clusters}\n"
        result_text += f"Noise points: {n_noise:,} ({n_noise / len(labels) * 100:.1f}%)\n\n"

        # Show cluster sizes (sorted by size)
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                size = np.sum(labels == label)
                cluster_sizes.append((label, size))

        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        result_text += "Cluster Details (sorted by size):\n"
        for i, (label, size) in enumerate(cluster_sizes[:20]):  # Show only first 20
            result_text += f"  Cluster {label:3d}: {size:6,} points\n"

        if len(cluster_sizes) > 20:
            result_text += f"  ... {len(cluster_sizes) - 20} more smaller clusters\n"

        result_text += f"\nResult file: {filepath}\n"
        result_text += f"{'=' * 50}\n"

        self.result_text.insert(tk.END, result_text)
        self.result_text.see(tk.END)

        self.status_var.set(f"Clustering completed - {n_clusters} clusters")
        self.current_result_file = filepath

        # Refresh preview with clustering results
        current_data = self.get_current_data()
        if current_data is not None:
            xyz_data = current_data[:, :3]
            self.refresh_clustering_preview(xyz_data, labels)

        # Reset progress bar
        self.progress_var.set(100)
        self.progress_label.config(text="Completed")

    def _clustering_error(self, error_msg):
        """Clustering error callback"""
        messagebox.showerror("Clustering Error", error_msg)
        self.status_var.set("Clustering failed")
        self.progress_var.set(0)
        self.progress_label.config(text="")

    def open_in_cc(self):
        """Open result in CloudCompare"""
        if not hasattr(self, 'current_result_file'):
            messagebox.showwarning("Warning", "Please execute clustering first")
            return

        success = self.cc_interface.open_in_cloudcompare(self.current_result_file)
        if success:
            self.status_var.set("Opened in CloudCompare")
        else:
            messagebox.showerror("Error", "Cannot start CloudCompare, please check path settings")

    def save_result(self):
        """Save current result as separate txt files"""
        if not hasattr(self, 'current_result_labels') or not hasattr(self, 'current_result_data'):
            messagebox.showwarning("Warning", "No clustering results to save, please execute clustering first")
            return

        # Choose save directory
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            self.status_var.set("Saving clustering results...")
            self.root.update()

            self._save_clustered_txt_files(
                self.current_result_data,
                self.current_result_labels,
                save_dir
            )
            messagebox.showinfo("Success", f"Clustering results saved to: {save_dir}")
            self.status_var.set("Save completed")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
            self.status_var.set("Save failed")

    def _save_clustered_txt_files(self, original_data, cluster_labels, save_dir):
        """Save clustering results as independent txt files (supports 10-column format)"""
        save_path = Path(save_dir)
        category = self.current_category

        # Create subdirectory
        output_subdir = save_path / f"{category}_dbscan_clusters"
        output_subdir.mkdir(exist_ok=True)

        # Get unique labels
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]
        noise_count = np.sum(cluster_labels == -1)

        saved_files = []
        cluster_stats = []

        # Save each valid cluster
        for i, label in enumerate(valid_labels):
            cluster_mask = cluster_labels == label
            cluster_data = original_data[cluster_mask]

            if len(cluster_data) > 0:
                # Filename format: category_cluster_number.txt
                filename = f"{category}_cluster_{i:03d}.txt"
                file_path = output_subdir / filename

                # Save as txt format (x y z r g b label nx ny nz)
                np.savetxt(file_path, cluster_data,
                           fmt='%.6f %.6f %.6f %d %d %d %d %.6f %.6f %.6f',
                           delimiter=' ')

                saved_files.append(str(file_path))
                cluster_stats.append({
                    'id': i,
                    'original_label': int(label),
                    'size': len(cluster_data),
                    'file': filename
                })

        # Save noise points (if any)
        if noise_count > 0:
            noise_mask = cluster_labels == -1
            noise_data = original_data[noise_mask]

            noise_filename = f"{category}_noise.txt"
            noise_file_path = output_subdir / noise_filename

            np.savetxt(noise_file_path, noise_data,
                       fmt='%.6f %.6f %.6f %d %d %d %d %.6f %.6f %.6f',
                       delimiter=' ')
            saved_files.append(str(noise_file_path))

        # Save statistics
        stats_file = output_subdir / f"{category}_statistics.txt"
        self._save_cluster_statistics(cluster_stats, noise_count, stats_file)

        # Update results display
        self._update_save_results_display(output_subdir, len(valid_labels), noise_count)

        return saved_files

    def _save_cluster_statistics(self, cluster_stats, noise_count, stats_file):
        """Save clustering statistics"""
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("DBSCAN Clustering Results Statistics\n")
            f.write("=" * 60 + "\n")
            f.write(f"Processing category: {self.current_category}\n")
            f.write(f"Clustering method: DBSCAN (Optimized)\n")
            f.write(f"Data format: x y z r g b label nx ny nz (10 columns)\n")
            f.write(f"Valid clusters: {len(cluster_stats)}\n")
            f.write(f"Noise points: {noise_count:,}\n")
            f.write(f"Total points: {sum(stat['size'] for stat in cluster_stats) + noise_count:,}\n")
            f.write("\nClustering parameters:\n")
            f.write(f"  eps: {self.param_vars['eps'].get()}\n")
            f.write(f"  min_samples: {self.param_vars['min_samples'].get()}\n")

            if self.use_optimization.get():
                f.write(f"  Optimization method: {self.optimization_method.get()}\n")
                if self.optimization_method.get() == 'voxel':
                    f.write(f"  Voxel size: {self.voxel_size_var.get()}\n")
                else:
                    f.write(f"  Batch size: {self.batch_size_var.get()}\n")

            f.write("\nDetailed information:\n")
            for stat in cluster_stats:
                f.write(f"Cluster {stat['id']:03d}: {stat['size']:6,} points -> {stat['file']}\n")

            if noise_count > 0:
                f.write(f"Noise     : {noise_count:6,} points -> noise.txt\n")

    def _update_save_results_display(self, output_dir, n_clusters, noise_count):
        """Update results display"""
        result_text = f"\n💾 Clustering results saved:\n"
        result_text += f"📁 Save directory: {output_dir}\n"
        result_text += f"📊 Valid clusters: {n_clusters}\n"
        result_text += f"🔸 Noise points: {noise_count:,}\n"
        result_text += f"📄 File format: txt (x y z r g b label nx ny nz)\n"
        result_text += f"📈 Statistics file: statistics.txt\n"

        self.result_text.insert(tk.END, result_text)
        self.result_text.see(tk.END)

    def save_config(self):
        """Save current parameter configuration"""
        save_path = filedialog.asksaveasfilename(
            title="Save Parameter Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if save_path:
            try:
                config = {
                    'method': 'DBSCAN',
                    'category': self.current_category,
                    'data_format': 'x_y_z_r_g_b_label_nx_ny_nz',
                    'parameters': {
                        'eps': self.param_vars['eps'].get(),
                        'min_samples': self.param_vars['min_samples'].get()
                    },
                    'optimization': {
                        'enabled': self.use_optimization.get(),
                        'method': self.optimization_method.get(),
                        'voxel_size': self.voxel_size_var.get(),
                        'batch_size': self.batch_size_var.get()
                    }
                }

                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("Success", f"Parameter configuration saved to: {save_path}")
                self.status_var.set("Configuration saved")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load parameter configuration"""
        config_path = filedialog.askopenfilename(
            title="Load Parameter Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Set category
                if 'category' in config:
                    self.category_var.set(config['category'])
                    self.on_category_change()

                # Set DBSCAN parameters
                if 'parameters' in config:
                    params = config['parameters']
                    if 'eps' in params:
                        self.param_vars['eps'].set(params['eps'])
                    if 'min_samples' in params:
                        self.param_vars['min_samples'].set(params['min_samples'])

                # Set optimization parameters
                if 'optimization' in config:
                    opt = config['optimization']
                    if 'enabled' in opt:
                        self.use_optimization.set(opt['enabled'])
                    if 'method' in opt:
                        self.optimization_method.set(opt['method'])
                    if 'voxel_size' in opt:
                        self.voxel_size_var.set(opt['voxel_size'])
                    if 'batch_size' in opt:
                        self.batch_size_var.set(opt['batch_size'])

                messagebox.showinfo("Success", f"Parameter configuration loaded: {config_path}")
                self.status_var.set("Configuration loaded")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def run(self):
        """Run GUI application"""
        self.root.mainloop()


# Program entry point
if __name__ == "__main__":
    """
    DBSCAN Clustering Tool - Large-scale Point Cloud Optimized

    Main Features:
    - Optimized DBSCAN algorithm supporting 100K+ point large-scale point clouds
    - Voxel downsampling for acceleration
    - Batch processing strategy
    - Real-time progress display
    - Real-time parameter adjustment
    - Result visualization and saving

    Optimization Strategies:
    1. Voxel downsampling: Downsample point cloud to voxel centers, greatly reducing computation
    2. Batch processing: Process large-scale point clouds in batches to avoid memory overflow
    3. Boundary refinement: Post-processing to optimize cluster boundaries
    4. Multi-threading: Use multi-core CPU for acceleration

    File Format:
    Input: x y z r g b label nx ny nz (10 columns)
    Output: Independent cluster files
    """
    try:
        app = RealtimeDBSCANTool()
        app.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install scipy scikit-learn numpy pandas")
    except Exception as e:
        print(f"Startup failed: {e}")
