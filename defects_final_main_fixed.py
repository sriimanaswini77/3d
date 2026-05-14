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
        self.root.title("Defect Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e2328')

        # ── Colour palette  (professional dark-steel theme) ─────────────
        self.BG        = '#1e2328'   # near-black charcoal
        self.PANEL     = '#252b33'   # dark steel panel
        self.ACCENT    = '#2e3a4a'   # muted slate-blue accent
        self.HIGHLIGHT = '#2d7dd2'   # professional azure blue
        self.FG        = '#dde3ea'   # soft white text
        self.FG2       = '#7f8c9a'   # muted grey-blue subtext
        self.BTN_BG    = '#2e3a4a'   # slate button
        self.BTN_ACT   = '#1a5fa8'   # darker azure on hover
        self.ENTRY_BG  = '#181d22'   # deep input background

        # ── ttk style ───────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('.',
                        background=self.BG,
                        foreground=self.FG,
                        fieldbackground=self.ENTRY_BG,
                        font=('Segoe UI', 10))

        style.configure('TFrame',       background=self.BG)
        style.configure('TLabel',       background=self.BG,  foreground=self.FG)
        style.configure('TLabelframe',  background=self.PANEL, foreground=self.FG)
        style.configure('TLabelframe.Label', background=self.PANEL, foreground=self.HIGHLIGHT,
                        font=('Segoe UI', 10, 'bold'))

        style.configure('TButton',
                        background=self.BTN_BG, foreground=self.FG,
                        font=('Segoe UI', 10, 'bold'), relief='flat',
                        borderwidth=0, padding=(10, 6))
        style.map('TButton',
                  background=[('active', self.BTN_ACT), ('disabled', '#2a3040')],
                  foreground=[('disabled', '#5a6472')])

        style.configure('Nav.TButton',
                        background=self.PANEL, foreground=self.FG2,
                        font=('Segoe UI', 11, 'bold'), relief='flat', padding=(16, 8))
        style.map('Nav.TButton',
                  background=[('active', self.HIGHLIGHT)],
                  foreground=[('active', '#ffffff')])

        style.configure('Active.Nav.TButton',
                        background=self.HIGHLIGHT, foreground='#ffffff',
                        font=('Segoe UI', 11, 'bold'), relief='flat', padding=(16, 8))

        style.configure('Danger.TButton',
                        background='#1e3a2f', foreground='#5dbf8e',
                        font=('Segoe UI', 10, 'bold'), relief='flat', padding=(10, 6))
        style.map('Danger.TButton', background=[('active', '#14532d')])

        style.configure('TEntry',
                        fieldbackground=self.ENTRY_BG, foreground=self.FG,
                        insertcolor=self.FG, relief='flat')
        style.configure('TScrollbar', background=self.ACCENT,
                        troughcolor=self.BG, bordercolor=self.BG)
        style.configure('Horizontal.TProgressbar',
                        troughcolor=self.PANEL, background=self.HIGHLIGHT,
                        bordercolor=self.BG)
        style.configure('TCheckbutton',
                        background=self.PANEL, foreground=self.FG)
        style.map('TCheckbutton', background=[('active', self.PANEL)])
        style.configure('TScale', background=self.PANEL, troughcolor=self.ACCENT)
        style.configure('TSpinbox',
                        fieldbackground=self.ENTRY_BG, foreground=self.FG,
                        background=self.ACCENT, arrowcolor=self.FG)
        style.configure('TSeparator', background=self.ACCENT)

        # ── Top header bar ──────────────────────────────────────────────
        header = tk.Frame(root, bg=self.PANEL, height=54)
        header.pack(side='top', fill='x')
        header.pack_propagate(False)

        # App title
        tk.Label(header, text='⬡  DEFECT DETECTION',
                 bg=self.PANEL, fg=self.HIGHLIGHT,
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=20)

        # Logout button (right-aligned) – hidden until login
        self._logout_btn = tk.Button(header, text='Logout',
                               bg='#1e3a2f', fg='#5dbf8e', relief='flat',
                               font=('Segoe UI', 10, 'bold'), padx=14, pady=4,
                               activebackground='#14532d', activeforeground='#ffffff',
                               cursor='hand2',
                               command=self._logout)
        # Do NOT pack yet — shown after login

        # Navigation tabs (horizontal pill buttons)
        nav_frame = tk.Frame(header, bg=self.PANEL)
        nav_frame.pack(side='left', padx=30)

        self._tab_buttons = {}
        self._current_tab = None

        tabs = [
            ('single',  'Single Image'),
            ('batch',   'Batch Process'),
            ('viz3d',   '3D Visualise'),
        ]
        for key, label in tabs:
            btn = tk.Button(nav_frame, text=label,
                            bg=self.PANEL, fg=self.FG2, relief='flat',
                            font=('Segoe UI', 11, 'bold'), padx=16, pady=6,
                            activebackground=self.HIGHLIGHT, activeforeground='#ffffff',
                            cursor='hand2',
                            command=lambda k=key: self._switch_tab(k))
            btn.pack(side='left', padx=2)
            self._tab_buttons[key] = btn

        # Hide Single Image tab until login
        self._tab_buttons['single'].pack_forget()

        # ── Content area ────────────────────────────────────────────────
        self.content = tk.Frame(root, bg=self.BG)
        self.content.pack(fill='both', expand=True)

        self.tab_frames = {}
        for key, _ in tabs:
            f = tk.Frame(self.content, bg=self.BG)
            self.tab_frames[key] = f

        # ── Footer ──────────────────────────────────────────────────────
        footer = tk.Frame(root, bg=self.PANEL, height=28)
        footer.pack(side='bottom', fill='x')
        footer.pack_propagate(False)
        tk.Label(footer, text='Designed and Developed by ABC',
                 bg=self.PANEL, fg=self.FG2, font=('Segoe UI', 9)).pack(expand=True)

        # ── Wire up legacy tab references ───────────────────────────────
        # Tab 1 = single image  (used by init_tab1 / watershed logic)
        self.tab2 = self.tab_frames['single']
        # Tab 3 = batch processing
        self.tab3 = self.tab_frames['batch']

        # ── Initialise tab contents ─────────────────────────────────────
        self.init_tab1()   # ROI Selection  → embedded inside 'single' tab
        self.init_tab2()   # Watershed      → uses self.tab2 (== 'single' frame)
        self.init_tab3()   # Batch          → uses self.tab3 (== 'batch'  frame)
        self._init_tab_viz3d()

        # Show default tab
        self._switch_tab('batch')

        # ── Login state & Ctrl+A+B shortcut ────────────────────────────
        self._logged_in = False
        self._keys_down = set()
        self.root.bind('<KeyPress>',   self._track_key_down)
        self.root.bind('<KeyRelease>', self._track_key_up)

    # ── Navigation helpers ──────────────────────────────────────────────
    def _switch_tab(self, key):
        # Single Image tab requires login
        if key == 'single' and not self._logged_in:
            self._open_login_popup()
            return
        for k, f in self.tab_frames.items():
            f.pack_forget()
        self.tab_frames[key].pack(fill='both', expand=True)
        for k, btn in self._tab_buttons.items():
            if k == key:
                btn.configure(bg=self.HIGHLIGHT, fg='#ffffff')
            else:
                btn.configure(bg=self.PANEL, fg=self.FG2)
        self._current_tab = key

    def _logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self._logged_in = False
            # Hide Single Image tab button and logout button
            self._tab_buttons['single'].pack_forget()
            self._logout_btn.pack_forget()
            # Switch back to batch tab
            self._switch_tab('batch')

    # ── Ctrl+A+B key tracking ───────────────────────────────────────────
    def _track_key_down(self, event):
        self._keys_down.add(event.keysym.lower())
        ctrl_held = ('control_l' in self._keys_down or
                     'control_r' in self._keys_down or
                     'control'   in self._keys_down)
        if ctrl_held and 'a' in self._keys_down and 'b' in self._keys_down:
            self._keys_down.clear()          # prevent repeated triggers
            self._open_login_popup()

    def _track_key_up(self, event):
        self._keys_down.discard(event.keysym.lower())

    # ── Login popup (triggered by Ctrl+A+B or tab guard) ───────────────
    def _open_login_popup(self):
        """Show a modal login dialog over the main window."""
        # If already logged in just switch to single tab directly
        if self._logged_in:
            self._switch_tab_direct('single')
            return

        popup = tk.Toplevel(self.root)
        popup.title("Login")
        popup.resizable(False, False)
        popup.configure(bg=self.BG)
        popup.grab_set()        # modal

        # Centre over main window
        self.root.update_idletasks()
        rx = self.root.winfo_x() + (self.root.winfo_width()  - 340) // 2
        ry = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        popup.geometry(f"340x400+{rx}+{ry}")

        PANEL = self.PANEL; BLUE = self.HIGHLIGHT
        FG = self.FG; FG2 = self.FG2; ENTRY = self.ENTRY_BG

        card = tk.Frame(popup, bg=PANEL)
        card.place(relx=0.5, rely=0.5, anchor='center', width=300, height=360)

        tk.Label(card, text='⬡', bg=PANEL, fg=BLUE,
                 font=('Segoe UI', 32)).pack(pady=(28, 0))
        tk.Label(card, text='SINGLE IMAGE ACCESS', bg=PANEL, fg=FG,
                 font=('Segoe UI', 11, 'bold')).pack()
        tk.Label(card, text='Login required', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9)).pack(pady=(2, 20))

        tk.Label(card, text='Username', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9), anchor='w').pack(fill='x', padx=30)
        user_var = tk.StringVar()
        user_entry = tk.Entry(card, textvariable=user_var,
                              bg=ENTRY, fg=FG, relief='flat',
                              font=('Segoe UI', 11), insertbackground=FG, bd=0)
        user_entry.pack(fill='x', padx=30, ipady=7, pady=(3, 12))

        tk.Label(card, text='Password', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9), anchor='w').pack(fill='x', padx=30)
        pass_var = tk.StringVar()
        pass_entry = tk.Entry(card, textvariable=pass_var, show='●',
                              bg=ENTRY, fg=FG, relief='flat',
                              font=('Segoe UI', 11), insertbackground=FG, bd=0)
        pass_entry.pack(fill='x', padx=30, ipady=7, pady=(3, 6))

        err_lbl = tk.Label(card, text='', bg=PANEL, fg='#e05c5c',
                           font=('Segoe UI', 9))
        err_lbl.pack(pady=(0, 10))

        def _try():
            if (user_var.get().strip() == 'admin' and
                    pass_var.get().strip() == 'admin'):
                self._logged_in = True
                popup.destroy()
                # Reveal Single Image tab button and logout button
                self._tab_buttons['single'].pack(side='left', padx=2)
                self._logout_btn.pack(side='right', padx=16, pady=10)
                self._switch_tab_direct('single')
            else:
                err_lbl.config(text='Invalid username or password.')
                pass_var.set('')

        tk.Button(card, text='Login', command=_try,
                  bg=BLUE, fg='#ffffff', relief='flat',
                  font=('Segoe UI', 11, 'bold'), padx=20, pady=7,
                  activebackground=self.BTN_ACT, cursor='hand2').pack(
                  fill='x', padx=30)

        user_entry.bind('<Return>', lambda e: pass_entry.focus())
        pass_entry.bind('<Return>', lambda e: _try())
        user_entry.focus()

    def _switch_tab_direct(self, key):
        """Switch tab without the login gate (used after successful login)."""
        for k, f in self.tab_frames.items():
            f.pack_forget()
        self.tab_frames[key].pack(fill='both', expand=True)
        for k, btn in self._tab_buttons.items():
            btn.configure(bg=self.HIGHLIGHT if k == key else self.PANEL,
                          fg='#ffffff' if k == key else self.FG2)
        self._current_tab = key

    # ── 3D Visualise tab ────────────────────────────────────────────────
    def _init_tab_viz3d(self):
        frame = self.tab_frames['viz3d']
        frame.configure(bg=self.BG)

        # Card panel — taller to fit Slicer path row
        card = tk.Frame(frame, bg=self.PANEL, bd=0, relief='flat')
        card.place(relx=0.5, rely=0.5, anchor='center', width=560, height=360)

        tk.Label(card, text='3D Visualise', bg=self.PANEL, fg=self.HIGHLIGHT,
                 font=('Segoe UI', 18, 'bold')).pack(pady=(24, 4))
        tk.Label(card, text='Image Source', bg=self.PANEL, fg=self.FG2,
                 font=('Segoe UI', 10)).pack(pady=(0, 8))

        self.viz3d_source = tk.StringVar(value='raw')

        for val, txt in [('raw',      'Raw Images from folder (as-is)'),
                         ('pipeline', 'Pipeline output images (processed results)')]:
            rb = tk.Radiobutton(card, text=txt, variable=self.viz3d_source, value=val,
                                bg=self.PANEL, fg=self.FG, selectcolor=self.ACCENT,
                                activebackground=self.PANEL, activeforeground=self.FG,
                                font=('Segoe UI', 10))
            rb.pack(anchor='w', padx=40, pady=2)

        sep1 = tk.Frame(card, bg=self.ACCENT, height=1)
        sep1.pack(fill='x', padx=30, pady=14)

        # ── Slicer executable path (hardcoded relative to project folder) ─
        # Project folder = directory containing this script.
        # Slicer is expected at:  <project_folder>/slicer/Slicer   (Linux/macOS)
        #                      or <project_folder>\slicer\Slicer.exe (Windows)
        _project_root = Path(__file__).resolve().parent
        _slicer_name  = 'Slicer.exe' if os.name == 'nt' else 'Slicer'
        _hardcoded_slicer = str(_project_root / 'slicer' / _slicer_name)
        self._slicer_exe_var = tk.StringVar(value=_hardcoded_slicer)

        sep2 = tk.Frame(card, bg=self.ACCENT, height=1)
        sep2.pack(fill='x', padx=30, pady=14)

        # ── Images folder path ───────────────────────────────────────
        tk.Label(card, text='Images Folder', bg=self.PANEL, fg=self.FG2,
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=30)

        self._viz3d_folder = tk.StringVar(value='No folder selected')
        folder_row = tk.Frame(card, bg=self.PANEL)
        folder_row.pack(fill='x', padx=30, pady=(4, 16))

        folder_entry = tk.Entry(folder_row, textvariable=self._viz3d_folder,
                                bg=self.ENTRY_BG, fg=self.FG2, relief='flat',
                                font=('Segoe UI', 9), state='readonly',
                                readonlybackground=self.ENTRY_BG, bd=0)
        folder_entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 8))

        def _pick_viz_folder():
            p = filedialog.askdirectory(title='Select Images Folder')
            if p:
                self._viz3d_folder.set(p)

        ttk.Button(folder_row, text='Upload Images Folder',
                   command=_pick_viz_folder).pack(side='right')

        btn_row = tk.Frame(card, bg=self.PANEL)
        btn_row.pack(pady=10)

        def _launch_3d():
            import glob, subprocess, tempfile, textwrap

            # ── Validate Slicer executable path ────────────────────────
            slicer_exe = self._slicer_exe_var.get().strip()
            if not slicer_exe or not os.path.isfile(slicer_exe):
                messagebox.showwarning(
                    '3D Slicer Not Found',
                    f'Slicer executable not found at the expected path:\n\n'
                    f'  {slicer_exe}\n\n'
                    'Please ensure the  slicer/  folder exists inside your project folder\n'
                    'and contains the Slicer executable.'
                )
                return

            # ── Validate images folder ──────────────────────────────────
            folder = self._viz3d_folder.get()
            if folder == 'No folder selected' or not os.path.isdir(folder):
                messagebox.showwarning('3D Viewer', 'Please select a valid images folder first.')
                return

            source = self.viz3d_source.get()

            # ── Collect image files ─────────────────────────────────────
            exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff',
                    '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIF', '*.TIFF')
            image_files = []
            for ext in exts:
                image_files.extend(glob.glob(os.path.join(folder, ext)))
            image_files = sorted(set(image_files))

            if not image_files:
                messagebox.showwarning('3D Viewer', 'No image files found in the selected folder.')
                return

            # ── Write Slicer Python startup script ─────────────────────
            # Key fix: connect to slicer.app.startupCompleted() signal so
            # ALL loading runs only after Slicer's main window is fully open.
            # Running any slicer.* calls before startupCompleted fires causes
            # a crash and only the splash logo is ever shown.
            script_content = textwrap.dedent(f"""
                import slicer
                import vtk
                import os

                IMAGE_FILES = {repr(image_files)}

                def load_data():
                    try:
                        # ── Switch to 3D-only layout immediately ──────────
                        layoutMgr = slicer.app.layoutManager()
                        layoutMgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

                        # Hide all slice viewers so only the 3D window is visible
                        for name in ('Red', 'Green', 'Yellow'):
                            w = layoutMgr.sliceWidget(name)
                            if w:
                                w.hide()

                        print("Loading {{}} images...".format(len(IMAGE_FILES)))
                        nodes = []
                        for fpath in IMAGE_FILES:
                            try:
                                n = slicer.util.loadVolume(fpath, properties={{'singleFile': True}})
                                if n:
                                    nodes.append(n)
                                    print("Loaded:", fpath)
                            except Exception as e:
                                print("Skipped", fpath, ":", e)

                        if not nodes:
                            print("Failed to load any nodes.")
                            return

                        if len(nodes) > 1:
                            appender = vtk.vtkImageAppend()
                            appender.SetAppendAxis(2)
                            for n in nodes:
                                appender.AddInputData(n.GetImageData())
                            appender.Update()
                            volNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'DefectVolume')
                            volNode.SetAndObserveImageData(appender.GetOutput())
                            volNode.CreateDefaultDisplayNodes()
                            for n in nodes:
                                slicer.mrmlScene.RemoveNode(n)
                        else:
                            volNode = nodes[0]

                        # ── Volume rendering ──────────────────────────────
                        logic = slicer.modules.volumerendering.logic()
                        displayNode = logic.CreateDefaultVolumeRenderingDisplayNode()
                        slicer.mrmlScene.AddNode(displayNode)
                        volNode.AddAndObserveDisplayNodeID(displayNode.GetID())
                        logic.UpdateDisplayNodeFromVolumeNode(displayNode, volNode)
                        displayNode.SetVisibility(True)

                        # Ensure layout stays at 3D-only after loading
                        layoutMgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                        slicer.util.resetThreeDViews()
                        print("Load complete — 3D render window active.")

                    except Exception as e:
                        import traceback
                        print("Error:", e)
                        print(traceback.format_exc())

                if slicer.app.mainWidgetPresent():
                    load_data()
                else:
                    slicer.app.connect("startupCompleted()", load_data)
            """)

            tmp_script = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False,
                prefix='defect_slicer_', encoding='utf-8')
            tmp_script.write(script_content)
            tmp_script.close()

            # ── Launch Slicer ───────────────────────────────────────────
            try:
                # Build a clean environment so cv2's bundled Qt plugins
                # do NOT interfere with Slicer's own Qt installation.
                env = os.environ.copy()

                # Remove Qt plugin paths injected by cv2 / anaconda
                for _key in ('QT_QPA_PLATFORM_PLUGIN_PATH',
                             'QT_PLUGIN_PATH',
                             'QT_DEBUG_PLUGINS'):
                    env.pop(_key, None)

                # Point to Slicer's own Qt plugins directory
                _slicer_root = os.path.dirname(os.path.dirname(os.path.abspath(slicer_exe)))
                for _candidate in (
                    os.path.join(_slicer_root, 'lib', 'Qt', 'plugins'),
                    os.path.join(_slicer_root, 'lib', 'qt5', 'plugins'),
                    os.path.join(_slicer_root, 'plugins'),
                ):
                    if os.path.isdir(_candidate):
                        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = _candidate
                        break

                # Force X11 backend — avoids Wayland/XDG_SESSION_TYPE crash
                env['QT_QPA_PLATFORM'] = 'xcb'
                env['XDG_SESSION_TYPE'] = 'x11'

                subprocess.Popen(
                    [slicer_exe,
                     '--no-splash',
                     '--python-script', tmp_script.name],
                    shell=False,
                    env=env
                )

                messagebox.showinfo(
                    '3D Viewer Launching',
                    f'3D Slicer is opening…\n\n'
                    f'Executable : {slicer_exe}\n'
                    f'Folder     : {folder}\n'
                    f'Source     : {source}\n'
                    f'Images     : {len(image_files)} file(s)\n\n'
                    'Slicer may take 10–30 seconds to start.\n'
                    'The volume will appear in the 3-D render window.\n\n'
                    'If nothing appears, open  View → Python Interactor\n'
                    'inside Slicer and check for error messages.'
                )
            except Exception as exc:
                messagebox.showerror('Launch Error',
                                     f'Failed to launch 3D Slicer:\n{exc}')

        tk.Button(btn_row, text='Launch 3D Viewer',
                  bg=self.HIGHLIGHT, fg='#ffffff', relief='flat',
                  font=('Segoe UI', 11, 'bold'), padx=20, pady=8,
                  activebackground=self.BTN_ACT, cursor='hand2',
                  command=_launch_3d).pack()

        # Controls cheatsheet
        ctrl = tk.Frame(card, bg=self.PANEL)
        ctrl.pack(fill='x', padx=30, pady=(10, 20))
        cheat = ('LEFT-DRAG: Rotate  |  SCROLL: Zoom  |  CLICK: Select panel  |  '
                 'ARROWS: Navigate  |  ENTER: Zoom in  |  ESC/R: Zoom out  |  '
                 'Q/E: Rotate 90°  |  H/V: Flip panel')
        tk.Label(ctrl, text=cheat, bg=self.PANEL, fg=self.FG2,
                 font=('Segoe UI', 8), wraplength=430, justify='center').pack()

    # ==================== TAB 1: ROI SELECTION (popup window) ====================

    def open_roi_window(self):
        """Open ROI Selection as a top-level window."""
        if self._roi_win is not None and self._roi_win.winfo_exists():
            self._roi_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("ROI Selection")
        win.geometry("900x650")
        win.configure(bg=self.BG)
        self._roi_win = win

        ctrl = tk.Frame(win, bg=self.PANEL, height=44)
        ctrl.pack(side='top', fill='x')
        ctrl.pack_propagate(False)

        for txt, cmd in [("Upload Image",     self.tab1_upload_image),
                         ("Clear Points",     self.tab1_clear_points),
                         ("Save Coordinates", self.tab1_save_coordinates)]:
            tk.Button(ctrl, text=txt, command=cmd,
                      bg=self.BTN_BG, fg=self.FG, relief='flat',
                      font=('Segoe UI', 9, 'bold'), padx=10, pady=4,
                      activebackground=self.HIGHLIGHT, cursor='hand2').pack(side='left', padx=6, pady=6)

        self.tab1_info_label = tk.Label(ctrl,
            text="Click on image to select polygon points (click on first point to close)",
            bg=self.PANEL, fg=self.FG2, font=('Segoe UI', 9))
        self.tab1_info_label.pack(side='left', padx=16)

        canvas_frame = tk.Frame(win, bg=self.BG)
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.tab1_canvas = tk.Canvas(canvas_frame, bg='#181d22', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical',
                                    command=self.tab1_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal',
                                    command=self.tab1_canvas.xview)
        self.tab1_canvas.configure(yscrollcommand=v_scrollbar.set,
                                   xscrollcommand=h_scrollbar.set)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        self.tab1_canvas.pack(side='left', fill='both', expand=True)

        self.tab1_canvas.bind('<Button-1>', self.tab1_canvas_click)
        self.tab1_canvas.bind('<MouseWheel>',
                              lambda e: self.tab1_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab1_canvas.bind('<Shift-MouseWheel>',
                              lambda e: self.tab1_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab1_canvas.bind('<Control-MouseWheel>', self.tab1_zoom)
        self.tab1_canvas.bind('<plus>',  lambda e: self.tab1_zoom_button(1.1))
        self.tab1_canvas.bind('<minus>', lambda e: self.tab1_zoom_button(0.9))
        self.tab1_canvas.bind('<equal>', lambda e: self.tab1_zoom_button(1.1))

        if self.tab1_image is not None:
            self.tab1_display_image()

    def init_tab1(self):
        """Initialise ROI Selection variables (UI is opened via open_roi_window)."""
        self.tab1_image = None
        self.tab1_cv_image = None
        self.tab1_photo = None
        self.tab1_polygon_points = []
        self.tab1_image_path = None
        self.tab1_polygon_closed = False
        self.tab1_zoom_level = 1.0
        self._roi_win = None
        # Placeholder label – real UI lives in open_roi_window()
        self.tab1_info_label = tk.Label(self.root)  # dummy; replaced when window opens
        

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

    # ==================== TAB 2: WATERSHED ANALYSIS (Single Image) ====================

    def init_tab2(self):
        """Initialize Single Image tab – Watershed Analysis with new GUI layout."""
        # Variables
        self.tab2_image = None
        self.tab2_cv_image = None
        self.tab2_roi_points = None
        self.tab2_image_path = None
        self.current_step = 0
        self.step_images = {}
        self.step_data = {}
        self.tab2_zoom_level = 1.0

        # Step 1 inline ROI drawing state
        self.step1_polygon_points = []
        self.step1_polygon_closed = False
        self._roi_save_dir = None

        # Default parameters for each step
        self.params = {
            'step_2_rotate_flip': {'enabled': True, 'rotate': 0, 'flip_h': False, 'flip_v': False},
            'step_3_grayscale': {'enabled': True},
            'step_4_sobel': {'enabled': True},
            'step_5_markers': {'enabled': True, 'low_threshold': 30, 'high_threshold': 150},
            'step_6_watershed': {'enabled': True},
            'step_7_morph_close': {'enabled': True, 'iterations': 2},
            'step_8_morph_open': {'enabled': True, 'iterations': 2},
            'step_9_fillholes': {'enabled': True},
            'step_10_labeling': {'enabled': True, 'min_size': 0},
            'step_11_final': {'min_area': 0, 'max_area': 10000},
            'step_12_perspective': {'enabled': True}
        }

        frame = self.tab2  # == self.tab_frames['single']
        frame.configure(bg=self.BG)

        # ── Top toolbar ──────────────────────────────────────────────────
        toolbar = tk.Frame(frame, bg=self.PANEL, height=46)
        toolbar.pack(side='top', fill='x')
        toolbar.pack_propagate(False)

        def _mk_btn(parent, text, cmd):
            return tk.Button(parent, text=text, command=cmd,
                             bg=self.BTN_BG, fg=self.FG, relief='flat',
                             font=('Segoe UI', 9, 'bold'), padx=10, pady=4,
                             activebackground=self.HIGHLIGHT, cursor='hand2')

        _mk_btn(toolbar, 'Load Image',       self.tab2_upload_image    ).pack(side='left', padx=6, pady=7)
        _mk_btn(toolbar, 'Load Config',      self.tab2_load_config     ).pack(side='left', padx=2, pady=7)
        _mk_btn(toolbar, 'Save Config',      self.tab2_save_config     ).pack(side='left', padx=2, pady=7)
        _mk_btn(toolbar, 'Start Processing', self.tab2_start_processing).pack(side='left', padx=2, pady=7)

        self.tab2_info_label = tk.Label(toolbar,
            text="ORIGINAL IMAGE: Raw loaded greyscale image – no processing applied yet",
            bg=self.PANEL, fg=self.FG2, font=('Segoe UI', 9))
        self.tab2_info_label.pack(side='left', padx=14)

        # ── Step number pills [1][2]…[10] ───────────────────────────────
        step_bar = tk.Frame(frame, bg=self.BG)
        step_bar.pack(side='top', fill='x', padx=8, pady=(6, 2))

        self._step_pill_btns = {}
        for i in range(13):
            label = str(i) if i > 0 else '0'
            btn = tk.Button(step_bar, text=label, width=3,
                            bg=self.ACCENT, fg=self.FG2, relief='flat',
                            font=('Segoe UI', 9, 'bold'),
                            activebackground=self.HIGHLIGHT,
                            cursor='hand2',
                            command=lambda s=i: self._jump_to_step(s))
            btn.pack(side='left', padx=2)
            self._step_pill_btns[i] = btn

        # ── Main content (image left, params right) ──────────────────────
        content = tk.Frame(frame, bg=self.BG)
        content.pack(fill='both', expand=True, padx=8, pady=(2, 4))

        # Left: image canvas
        img_frame = tk.Frame(content, bg=self.BG)
        img_frame.pack(side='left', fill='both', expand=True, padx=(0, 6))

        self.tab2_canvas = tk.Canvas(img_frame, bg='#181d22', highlightthickness=0)
        v_sb = ttk.Scrollbar(img_frame, orient='vertical', command=self.tab2_canvas.yview)
        h_sb = ttk.Scrollbar(img_frame, orient='horizontal', command=self.tab2_canvas.xview)
        self.tab2_canvas.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        v_sb.pack(side='right', fill='y')
        h_sb.pack(side='bottom', fill='x')
        self.tab2_canvas.pack(side='left', fill='both', expand=True)

        self.tab2_canvas.bind('<MouseWheel>',
                              lambda e: self.tab2_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab2_canvas.bind('<Shift-MouseWheel>',
                              lambda e: self.tab2_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        self.tab2_canvas.bind('<Control-MouseWheel>', self.tab2_zoom)
        # Linux uses Button-4 (scroll up) and Button-5 (scroll down)
        self.tab2_canvas.bind('<Control-Button-4>', self.tab2_zoom)
        self.tab2_canvas.bind('<Control-Button-5>', self.tab2_zoom)
        self.tab2_canvas.bind('<plus>',  lambda e: self.tab2_zoom_button(1.1))
        self.tab2_canvas.bind('<minus>', lambda e: self.tab2_zoom_button(0.9))
        self.tab2_canvas.bind('<equal>', lambda e: self.tab2_zoom_button(1.1))
        self.tab2_canvas.bind('<Button-1>', self.tab2_step1_canvas_click)
        # Ctrl+drag to zoom
        self.tab2_canvas.bind('<Control-ButtonPress-1>',  self._tab2_drag_zoom_start)
        self.tab2_canvas.bind('<Control-B1-Motion>',      self._tab2_drag_zoom_motion)

        # Placeholder label when no image loaded
        self._img_placeholder = tk.Label(self.tab2_canvas,
            text='▲\nLoad Image to begin',
            bg='#181d22', fg='#3e4e5e',
            font=('Segoe UI', 14), justify='center')
        self._img_placeholder.place(relx=0.5, rely=0.5, anchor='center')

        # Right: parameters panel
        right_panel = tk.Frame(content, bg=self.PANEL, width=320)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)

        # Step info inside right panel
        step_info_frame = tk.Frame(right_panel, bg=self.PANEL)
        step_info_frame.pack(fill='x', padx=10, pady=(12, 4))

        self.step_name_label = tk.Label(step_info_frame, text="Step 0: Original Image",
                                        bg=self.PANEL, fg=self.HIGHLIGHT,
                                        font=('Segoe UI', 11, 'bold'), wraplength=280, justify='left')
        self.step_name_label.pack(anchor='w')

        self.step_desc_label = tk.Label(step_info_frame, text="Load an image to begin",
                                        bg=self.PANEL, fg=self.FG2,
                                        font=('Segoe UI', 9), wraplength=280, justify='left')
        self.step_desc_label.pack(anchor='w', pady=(4, 0))

        sep = tk.Frame(right_panel, bg=self.ACCENT, height=1)
        sep.pack(fill='x', padx=10, pady=8)

        # Parameters header
        tk.Label(right_panel, text='Parameters',
                 bg=self.PANEL, fg=self.FG,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=12)

        # Scrollable params canvas
        params_outer = tk.Frame(right_panel, bg=self.PANEL)
        params_outer.pack(fill='both', expand=True, padx=6, pady=6)

        self.params_canvas = tk.Canvas(params_outer, bg=self.PANEL, highlightthickness=0)
        params_scrollbar = ttk.Scrollbar(params_outer, orient='vertical',
                                         command=self.params_canvas.yview)
        self.params_frame = tk.Frame(self.params_canvas, bg=self.PANEL)

        self.params_canvas.configure(yscrollcommand=params_scrollbar.set)
        params_scrollbar.pack(side='right', fill='y')
        self.params_canvas.pack(side='left', fill='both', expand=True)

        self.params_canvas_window = self.params_canvas.create_window(
            (0, 0), window=self.params_frame, anchor='nw')

        self.params_frame.bind('<Configure>',
                               lambda e: self.params_canvas.configure(
                                   scrollregion=self.params_canvas.bbox('all')))

        # ── Bottom navigation (Prev / Next) ─────────────────────────────
        nav_bar = tk.Frame(frame, bg=self.PANEL, height=44)
        nav_bar.pack(side='bottom', fill='x')
        nav_bar.pack_propagate(False)

        self.progress_label = tk.Label(nav_bar, text="Step 0 / 12",
                                       bg=self.PANEL, fg=self.FG2,
                                       font=('Segoe UI', 10))
        self.progress_label.pack(side='left', padx=16)

        # ── Zoom controls in nav bar ─────────────────────────────────────
        zoom_frame = tk.Frame(nav_bar, bg=self.PANEL)
        zoom_frame.pack(side='left', padx=20)

        tk.Label(zoom_frame, text='Zoom:', bg=self.PANEL, fg=self.FG2,
                 font=('Segoe UI', 9)).pack(side='left', padx=(0, 4))

        tk.Button(zoom_frame, text='−', width=3,
                  command=lambda: self.tab2_zoom_button(0.8),
                  bg=self.BTN_BG, fg=self.FG, relief='flat',
                  font=('Segoe UI', 11, 'bold'), pady=2,
                  activebackground=self.HIGHLIGHT, cursor='hand2').pack(side='left', padx=2)

        tk.Button(zoom_frame, text='+', width=3,
                  command=lambda: self.tab2_zoom_button(1.25),
                  bg=self.BTN_BG, fg=self.FG, relief='flat',
                  font=('Segoe UI', 11, 'bold'), pady=2,
                  activebackground=self.HIGHLIGHT, cursor='hand2').pack(side='left', padx=2)

        tk.Button(zoom_frame, text='Reset', width=5,
                  command=self.tab2_zoom_reset,
                  bg=self.BTN_BG, fg=self.FG, relief='flat',
                  font=('Segoe UI', 9, 'bold'), pady=2,
                  activebackground=self.HIGHLIGHT, cursor='hand2').pack(side='left', padx=2)

        self.zoom_label = tk.Label(zoom_frame, text='100%', bg=self.PANEL, fg=self.FG2,
                                   font=('Segoe UI', 9), width=5)
        self.zoom_label.pack(side='left', padx=(4, 0))

        self.next_button = tk.Button(nav_bar, text='Next  ›',
                                     command=self.tab2_next_step,
                                     bg=self.HIGHLIGHT, fg='#ffffff', relief='flat',
                                     font=('Segoe UI', 10, 'bold'), padx=14, pady=6,
                                     activebackground=self.BTN_ACT, cursor='hand2',
                                     state='disabled')
        self.next_button.pack(side='right', padx=10, pady=6)

        self.prev_button = tk.Button(nav_bar, text='‹  Previous',
                                     command=self.tab2_previous_step,
                                     bg=self.BTN_BG, fg=self.FG, relief='flat',
                                     font=('Segoe UI', 10, 'bold'), padx=14, pady=6,
                                     activebackground=self.HIGHLIGHT, cursor='hand2',
                                     state='disabled')
        self.prev_button.pack(side='right', padx=4, pady=6)

        self.tab2_photo = None

    def _jump_to_step(self, step):
        """Jump to a specific step when a pill button is clicked."""
        if step in self.step_images:
            self.current_step = step
            self.display_current_step()

    def _update_step_pills(self):
        """Highlight the currently active step pill."""
        for i, btn in self._step_pill_btns.items():
            if i == self.current_step:
                btn.configure(bg=self.HIGHLIGHT, fg='#ffffff')
            elif i in self.step_images or (i == 1 and len(self.step1_polygon_points) > 0):
                btn.configure(bg=self.ACCENT, fg=self.FG)
            else:
                btn.configure(bg=self.ACCENT, fg=self.FG2)

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
        """Zoom image with Ctrl+MouseWheel (Windows/macOS: delta; Linux: num)"""
        if self.tab2_image is None:
            return
        # event.delta works on Windows/macOS; on Linux Button-4/5 give delta=0
        if event.delta != 0:
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            # Linux scroll up = Button-4, scroll down = Button-5
            factor = 1.1 if getattr(event, 'num', 0) == 4 else 0.9
        self.tab2_zoom_level *= factor
        self.tab2_zoom_level = max(0.1, min(10.0, self.tab2_zoom_level))
        self._tab2_update_zoom_label()
        if self.current_step == 1:
            self.display_step1_drawing()
        elif self.current_step in self.step_images:
            self.display_current_step()

    def tab2_zoom_button(self, factor):
        """Zoom image with +/- buttons or keys"""
        if self.current_step != 1 and self.current_step not in self.step_images:
            return
        self.tab2_zoom_level *= factor
        self.tab2_zoom_level = max(0.1, min(10.0, self.tab2_zoom_level))
        self._tab2_update_zoom_label()
        self.display_current_step()
        
    def tab2_zoom_reset(self):
        """Reset zoom to 100%."""
        self.tab2_zoom_level = 1.0
        self._tab2_update_zoom_label()
        self.display_current_step()

    def _tab2_update_zoom_label(self):
        """Refresh the zoom % label in the nav bar."""
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"{int(self.tab2_zoom_level * 100)}%")

    def _tab2_drag_zoom_start(self, event):
        """Record start Y position for Ctrl+drag zoom."""
        self._drag_zoom_start_y = event.y
        self._drag_zoom_start_level = self.tab2_zoom_level

    def _tab2_drag_zoom_motion(self, event):
        """Ctrl+drag up = zoom in, drag down = zoom out."""
        if not hasattr(self, '_drag_zoom_start_y'):
            return
        delta = self._drag_zoom_start_y - event.y   # positive = dragged up
        factor = 1.0 + delta * 0.005                # 0.5% per pixel
        new_level = self._drag_zoom_start_level * factor
        self.tab2_zoom_level = max(0.1, min(10.0, new_level))
        self._tab2_update_zoom_label()
        if self.current_step != 1 and self.current_step not in self.step_images:
            return
        self.display_current_step()

    def tab2_step1_canvas_click(self, event):
        """Handle canvas click: only active during Step 1 ROI drawing."""
        if self.current_step != 1:
            return
        if self.tab2_image is None:
            return
        if self.step1_polygon_closed:
            return

        # Convert canvas coords → image coords (account for zoom)
        cx = int(self.tab2_canvas.canvasx(event.x))
        cy = int(self.tab2_canvas.canvasy(event.y))
        x = int(cx / self.tab2_zoom_level)
        y = int(cy / self.tab2_zoom_level)

        # Clamp to image bounds
        h, w = self.tab2_image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # Check if clicking near first point to close polygon
        if len(self.step1_polygon_points) >= 3:
            fp = self.step1_polygon_points[0]
            dist = math.sqrt((x - fp[0])**2 + (y - fp[1])**2)
            if dist <= int(15 / self.tab2_zoom_level):
                self.step1_polygon_closed = True
                self.tab2_roi_points = np.array(self.step1_polygon_points, dtype=np.int32)
                self._step1_auto_save()
                self._display_step1_drawing()
                return

        self.step1_polygon_points.append((x, y))
        self._display_step1_drawing()

    def _step1_auto_save(self):
        """Auto-save ROI coordinates JSON next to the loaded image.
           The config JSON (Save Config) will use the same folder."""
        if self.tab2_image_path is None:
            return
        img_dir  = os.path.dirname(os.path.abspath(self.tab2_image_path))
        img_stem = Path(self.tab2_image_path).stem
        roi_path = os.path.join(img_dir, f"{img_stem}_roi.json")

        data = {
            "image_path":  self.tab2_image_path,
            "image_shape": list(self.tab2_image.shape[:2]),
            "polygon_points": self.step1_polygon_points,
            "pipeline_params": self.params
        }
        with open(roi_path, 'w') as f:
            json.dump(data, f, indent=4)

        # ── Also write a copy to configurations/conf.json so Batch tab
        #    can auto-load it on next startup ──────────────────────────
        try:
            conf_dir = Path(__file__).resolve().parent / 'configurations'
            conf_dir.mkdir(exist_ok=True)
            with open(conf_dir / 'conf.json', 'w') as _cf:
                json.dump(data, _cf, indent=4)
        except Exception:
            pass  # never block the main save if this fails

        # Remember save folder so Save Config lands in same place
        self._roi_save_dir = img_dir
        self.tab2_info_label.config(
            text=f"ROI auto-saved → {roi_path}  |  Click Next to run pipeline.")
        messagebox.showinfo("ROI Saved",
            f"Polygon closed!\n\nCoordinates auto-saved to:\n{roi_path}\n\nClick Next to run the pipeline.")

    def tab2_step1_clear_points(self):
        """Clear Step 1 polygon drawing."""
        self.step1_polygon_points = []
        self.step1_polygon_closed = False
        self.tab2_roi_points = None
        self._display_step1_drawing()

    def tab2_start_processing(self):
        """Start the step-by-step watershed processing.

        Behaviour:
        - If the user has previously clicked 'Load Config' and the config
          contained polygon_points, those points are already stored in
          self.step1_polygon_points / self.step1_polygon_closed.
          → Step 1 will open with the saved polygon pre-drawn so the user
            can verify or modify it before proceeding.
        - If no config was loaded (or the config had no polygon), the Step 1
          polygon state is empty and the user draws manually as before.
        """
        if self.tab2_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return

        # ── Determine whether a pre-loaded ROI is available ──────────────
        # (populated by tab2_load_config when JSON contains polygon_points)
        roi_pre_loaded = (
            self.step1_polygon_closed and
            len(self.step1_polygon_points) >= 3
        )

        # If no config was loaded at all, make sure the drawing state is clean
        if not roi_pre_loaded:
            self.tab2_roi_points      = None
            self.step1_polygon_points = []
            self.step1_polygon_closed = False

        # Show Step 0: original image only
        self.step_images = {}
        self.step_images[0] = self.tab2_image.copy()
        self.current_step = 0
        self.display_current_step()

        # Allow moving to Step 1 (ROI draw/review)
        self.next_button.config(state='normal')
        self.prev_button.config(state='disabled')

        if roi_pre_loaded:
            self.tab2_info_label.config(
                text=f"✔ ROI from loaded config pre-filled "
                     f"({len(self.step1_polygon_points)} points). "
                     f"Click Next to review / edit in Step 1, then proceed.")
        else:
            self.tab2_info_label.config(
                text="Step 0 loaded. Click Next to draw ROI polygon in Step 1.")

    def process_all_steps(self):
        """Process all watershed steps using Scikit-Image approach (Sobel + Histogram Markers)"""
        self.step_images = {}
        self.step_data = {}

        # Step 0: Original Image
        self.step_images[0] = self.tab2_image.copy()

        # Step 1: RoI Masked Image (4-point polygon)
        mask = np.zeros(self.tab2_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.tab2_roi_points], 255)
        masked_image = self.tab2_cv_image.copy()
        masked_image[mask == 0] = [0, 0, 0]
        self.step_images[1] = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        self.step_data['mask'] = mask
        self.step_data['masked_bgr'] = masked_image

        # Step 2: Rotate / Flip  (NEW)
        rotated_bgr = masked_image.copy()
        if self.params['step_2_rotate_flip']['enabled']:
            angle = self.params['step_2_rotate_flip']['rotate']
            if angle != 0:
                h, w = rotated_bgr.shape[:2]
                cx, cy = w // 2, h // 2
                M_rot = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                rotated_bgr = cv2.warpAffine(rotated_bgr, M_rot, (w, h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
            if self.params['step_2_rotate_flip']['flip_h']:
                rotated_bgr = cv2.flip(rotated_bgr, 1)
            if self.params['step_2_rotate_flip']['flip_v']:
                rotated_bgr = cv2.flip(rotated_bgr, 0)
        self.step_images[2] = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
        self.step_data['rotated_bgr'] = rotated_bgr

        # Step 3: Grayscale Conversion  (was Step 2)
        if self.params['step_3_grayscale']['enabled']:
            gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
            self.step_images[3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
            self.step_images[3] = self.step_images[2].copy()
        self.step_data['gray'] = gray

        # Step 4: Sobel Gradient (Elevation Map)  (was Step 3)
        if self.params['step_4_sobel']['enabled']:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            elevation_map = np.hypot(sobel_x, sobel_y)
            elevation_display = cv2.normalize(elevation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            elevation_map = gray.astype(np.float64)
            elevation_display = gray
        self.step_images[4] = cv2.cvtColor(elevation_display, cv2.COLOR_GRAY2RGB)
        self.step_data['elevation_map'] = elevation_map

        # Step 5: Histogram-based Markers  (was Step 4)
        if self.params['step_5_markers']['enabled']:
            low_thresh  = self.params['step_5_markers']['low_threshold']
            high_thresh = self.params['step_5_markers']['high_threshold']

            inv = 255 - gray
            background = cv2.GaussianBlur(inv, (0, 0), sigmaX=50, sigmaY=50)
            gbImg = cv2.subtract(inv, background)
            _, gb = cv2.threshold(gbImg, 10, 255, cv2.THRESH_BINARY)

            markers = np.zeros_like(gray, dtype=np.int32)
            background_mask = gray < low_thresh
            markers[background_mask] = gb[background_mask]

            foreground_mask = gray > high_thresh
            markers[foreground_mask] = gray[foreground_mask].astype(np.int32) + 1

            markers_display = np.zeros_like(gray)
            markers_display[background_mask] = gb[background_mask]
            markers_display[foreground_mask] = gray[foreground_mask]
        else:
            markers = np.zeros_like(gray, dtype=np.int32)
            markers[gray > 0] = gray[gray > 0].astype(np.int32) + 1
            markers_display = gray.copy()

        self.step_images[5] = cv2.cvtColor(markers_display, cv2.COLOR_GRAY2RGB)
        self.step_data['markers'] = markers

        # Step 6: Watershed Application  (was Step 5)
        if self.params['step_6_watershed']['enabled']:
            elevation_map_uint8 = elevation_display
            elevation_map_color = cv2.cvtColor(elevation_map_uint8, cv2.COLOR_GRAY2BGR)
            markers_watershed = markers.copy()
            segmentation = cv2.watershed(elevation_map_color, markers_watershed)

            watershed_display = np.zeros_like(self.step_data['rotated_bgr'])
            watershed_display[segmentation == 1]  = [50, 50, 50]
            watershed_display[segmentation == 2]  = [0, 200, 0]
            watershed_display[segmentation == -1] = [255, 0, 0]
            watershed_display[segmentation == 0]  = [0, 0, 0]
        else:
            segmentation = markers.copy()
            watershed_display = self.step_data['rotated_bgr'].copy()

        self.step_images[6] = cv2.cvtColor(watershed_display, cv2.COLOR_BGR2RGB)
        self.step_data['segmentation'] = segmentation

        # Convert segmentation to binary
        if self.params['step_5_markers']['enabled']:
            high_thresh = self.params['step_5_markers']['high_threshold']
            binary_seg = (segmentation > high_thresh).astype(np.uint8)
        else:
            binary_seg = (segmentation > 0).astype(np.uint8)

        # Step 7: Morphological Close  (was Step 6)
        if self.params['step_7_morph_close']['enabled']:
            iterations = self.params['step_7_morph_close']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(binary_seg, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            morph_close = binary_seg.copy()

        self.step_images[7] = cv2.cvtColor(morph_close * 255, cv2.COLOR_GRAY2RGB)
        self.step_data['morph_close'] = morph_close

        # Step 8: Morphological Open  (was Step 7)
        if self.params['step_8_morph_open']['enabled']:
            iterations = self.params['step_8_morph_open']['iterations']
            kernel = np.ones((3, 3), np.uint8)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            morph_open = morph_close.copy()

        self.step_images[8] = cv2.cvtColor(morph_open * 255, cv2.COLOR_GRAY2RGB)
        self.step_data['morph_open'] = morph_open

        # Step 9: Fill Holes  (was Step 8)
        if self.params['step_9_fillholes']['enabled']:
            filled = ndi.binary_fill_holes(morph_open).astype(np.uint8) * 255
        else:
            filled = morph_open * 255

        self.step_images[9] = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)
        self.step_data['filled'] = filled

        # Step 10: Label Connected Components  (was Step 9)
        if self.params['step_10_labeling']['enabled']:
            labeled_array, num_features = ndi.label(filled)
            min_size = self.params['step_10_labeling']['min_size']
            sizes = np.bincount(labeled_array.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0
            cleaned_labels = mask_sizes[labeled_array]
            labeled_cleaned, num_cleaned = ndi.label(cleaned_labels)
            labeled_display = cv2.normalize(labeled_cleaned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            labeled_colored = cv2.applyColorMap(labeled_display, cv2.COLORMAP_JET)
        else:
            labeled_cleaned, num_cleaned = ndi.label(filled)
            labeled_display = cv2.normalize(labeled_cleaned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            labeled_colored = cv2.applyColorMap(labeled_display, cv2.COLORMAP_JET)

        self.step_images[10] = cv2.cvtColor(labeled_colored, cv2.COLOR_BGR2RGB)
        self.step_data['labeled'] = labeled_cleaned
        self.step_data['num_labels'] = num_cleaned

        # Step 11: Final Result with Colored Segments  (was Step 10)
        result = self.tab2_cv_image.copy()
        min_area = self.params['step_11_final']['min_area']
        max_area = self.params['step_11_final']['max_area']
        segment_color = (0, 0, 255)

        for region_label in range(1, self.step_data['num_labels'] + 1):
            region_mask = (labeled_cleaned == region_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area <= area <= max_area:
                        cv2.drawContours(result, [contour], -1, segment_color, -1)

        self.step_images[11] = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Step 12: Perspective Warp  (was Step 11)
        if self.params['step_12_perspective']['enabled'] and self.tab2_roi_points is not None:
            roi_pts = self.tab2_roi_points.reshape(-1, 2).astype(np.float32)
            s = roi_pts.sum(axis=1)
            d = np.diff(roi_pts, axis=1).ravel()
            tl = roi_pts[np.argmin(s)]
            br = roi_pts[np.argmax(s)]
            tr = roi_pts[np.argmin(d)]
            bl = roi_pts[np.argmax(d)]
            src_quad = np.array([tl, tr, br, bl], dtype=np.float32)
            out_w = 272
            out_h = 272
            dst_quad = np.array([[0, 0], [out_w - 1, 0],
                                 [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_quad, dst_quad)
            warped = cv2.warpPerspective(result, M, (out_w, out_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        else:
            warped = result.copy()

        self.step_images[12] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        self.step_data['warped'] = warped

    def display_current_step(self):
        """Display the current step's image and parameters"""
        # Step 1 is interactive: display it even before step_images[1] exists
        if self.current_step == 1:
            self._display_step1_drawing()
            return

        if self.current_step not in self.step_images:
            return

        # Update step info
        step_names = [
            "Step 0: Original Image",
            "Step 1: RoI Selection (4-Point Polygon)",
            "Step 2: Rotate / Flip",
            "Step 3: Grayscale Conversion",
            "Step 4: Sobel Gradient (Elevation Map)",
            "Step 5: Histogram Markers",
            "Step 6: Watershed Application",
            "Step 7: Morphological Close",
            "Step 8: Morphological Open",
            "Step 9: Fill Holes",
            "Step 10: Connected Component Labeling",
            "Step 11: Final Result",
            "Step 12: Perspective Warp"
        ]

        step_descriptions = [
            "Original loaded image without any processing.",
            "Image with Area of Interest applied. Regions outside the 4-point polygon are blacked out.",
            "Rotate and/or flip the masked image before further processing.",
            "Image converted to grayscale for processing.",
            "Sobel gradient (elevation map) computed. High gradient values form barriers between regions.",
            "Markers with varying values: background uses low intensities, foreground uses high intensities.",
            "Watershed algorithm floods elevation map from markers. Boundaries marked in red.",
            "Morphological closing to fill small holes and connect nearby objects.",
            "Morphological opening to remove small noise and separate touching objects.",
            "Binary fill holes removes small holes in detected foreground regions.",
            "Connected component labeling identifies individual objects. Small objects filtered out.",
            "Final result with detected segments colored in red.",
            "Perspective warp: ROI polygon corners mapped to a flat rectangle for top-down defect view."
        ]

        self.step_name_label.config(text=step_names[self.current_step])
        self.step_desc_label.config(text=step_descriptions[self.current_step])
        self.progress_label.config(text=f"Step {self.current_step} / 12")

        # Hide placeholder
        self._img_placeholder.place_forget()

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
        self.next_button.config(state='normal' if self.current_step < 12 else 'disabled')

        # Update step pills
        self._update_step_pills()

    def _display_step1_drawing(self):
        """Render Step 1 interactive ROI drawing on the main tab2 canvas."""
        step_names = ["Step 0: Original Image",
                      "Step 1: RoI Selection (Draw Polygon)",]
        self.step_name_label.config(text="Step 1: RoI Selection (Draw Polygon)")
        self.step_desc_label.config(
            text="Click on the image to place polygon points. "
                 "Click near the first point (green circle) to close the polygon.")
        self.progress_label.config(text="Step 1 / 12")
        self._img_placeholder.place_forget()

        # Draw original image with polygon overlay
        display_img = self.tab2_image.copy()
        pts = self.step1_polygon_points
        zoom = self.tab2_zoom_level

        if len(pts) > 0:
            # Draw lines between consecutive points
            for i in range(len(pts) - 1):
                p1 = (int(pts[i][0] * zoom), int(pts[i][1] * zoom))
                p2 = (int(pts[i+1][0] * zoom), int(pts[i+1][1] * zoom))
                cv2.line(display_img, pts[i], pts[i+1], (0, 255, 0), 2)
            if self.step1_polygon_closed:
                cv2.line(display_img, pts[-1], pts[0], (0, 255, 0), 2)
                # Fill polygon overlay semi-transparent style (solid tint)
                overlay = display_img.copy()
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, display_img, 0.85, 0, display_img)
                cv2.polylines(display_img, [pts_arr], True, (0, 255, 0), 2)
            # Draw point circles
            for idx, pt in enumerate(pts):
                color = (255, 215, 0) if idx == 0 else (255, 0, 0)
                cv2.circle(display_img, pt, 7, color, -1)
                cv2.circle(display_img, pt, 8, (255, 255, 255), 2)
                cv2.putText(display_img, str(idx + 1), (pt[0]+10, pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        pil_img = Image.fromarray(display_img)
        new_width  = int(pil_img.width  * self.tab2_zoom_level)
        new_height = int(pil_img.height * self.tab2_zoom_level)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tab2_photo = ImageTk.PhotoImage(pil_img)

        self.tab2_canvas.delete("all")
        self.tab2_canvas.create_image(0, 0, anchor='nw', image=self.tab2_photo)
        self.tab2_canvas.configure(scrollregion=self.tab2_canvas.bbox("all"))

        # Update right panel controls for step 1
        self.update_parameter_controls()

        # Nav buttons
        self.prev_button.config(state='normal')
        self.next_button.config(state='normal')
        self._update_step_pills()

        # Update info label
        if self.step1_polygon_closed:
            self.tab2_info_label.config(
                text=f"Polygon closed with {len(pts)} points. "
                     f"Coordinates auto-saved. Click Next to run pipeline.")
        elif len(pts) == 0:
            self.tab2_info_label.config(
                text="Step 1: Click on the image to place ROI polygon points.")
        elif len(pts) < 3:
            self.tab2_info_label.config(
                text=f"Points placed: {len(pts)}. Need at least 3 points.")
        else:
            self.tab2_info_label.config(
                text=f"Points placed: {len(pts)}. Click near point 1 (gold circle) to close polygon.")

    def update_parameter_controls(self):
        """Update parameter controls based on current step"""
        # Clear existing controls
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        step = self.current_step

        if step == 1:   # ROI Drawing
            self.create_step1_controls()
        elif step == 2:   # Rotate / Flip (NEW)
            self.create_step2_controls()
        elif step == 3:  # Grayscale
            self.create_step3_controls()
        elif step == 4:  # Sobel
            self.create_step4_controls()
        elif step == 5:  # Markers
            self.create_step5_controls()
        elif step == 6:  # Watershed
            self.create_step6_controls()
        elif step == 7:  # Morphological Close
            self.create_step7_controls()
        elif step == 8:  # Morphological Open
            self.create_step8_controls()
        elif step == 9:  # Fill Holes
            self.create_step9_controls()
        elif step == 10:  # Labeling
            self.create_step10_controls()
        elif step == 11:  # Final
            self.create_step11_controls()
        elif step == 12:  # Perspective Warp
            self.create_step12_controls()
        else:
            ttk.Label(self.params_frame, text="No adjustable parameters for this step.",
                     wraplength=280).pack(pady=10)

    def create_step1_controls(self):
        """Right-panel controls for Step 1: ROI Drawing."""
        tk.Label(self.params_frame,
                 text="Draw ROI Polygon",
                 bg=self.PANEL, fg=self.HIGHLIGHT,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(8, 4))

        tk.Label(self.params_frame,
                 text="1. Click on the image to place polygon points.\n"
                      "2. Point 1 is shown as a gold circle.\n"
                      "3. Click near point 1 to close the polygon.\n"
                      "4. Coordinates are auto-saved next to your image.\n"
                      "5. Click Next to run Steps 2-12.",
                 bg=self.PANEL, fg=self.FG2,
                 font=('Segoe UI', 9), wraplength=280, justify='left').pack(anchor='w', padx=4, pady=(0, 12))

        tk.Frame(self.params_frame, bg=self.ACCENT, height=1).pack(fill='x', pady=6)

        # Points counter
        n = len(self.step1_polygon_points)
        status = "Closed ✔" if self.step1_polygon_closed else f"{n} point(s) placed"
        self._step1_count_lbl = tk.Label(self.params_frame,
                 text=f"Status: {status}",
                 bg=self.PANEL, fg=self.FG,
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=4, pady=(0, 8))

        tk.Button(self.params_frame,
                  text="Clear Points",
                  command=self.tab2_step1_clear_points,
                  bg=self.BTN_BG, fg=self.FG, relief='flat',
                  font=('Segoe UI', 9, 'bold'), padx=10, pady=4,
                  activebackground=self.HIGHLIGHT, cursor='hand2').pack(anchor='w', padx=4)

    def create_step2_controls(self):
        """Create controls for Step 2: Rotate / Flip"""
        skip_var = tk.BooleanVar(value=not self.params['step_2_rotate_flip']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_2_rotate_flip', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        # Rotation angle
        ttk.Label(self.params_frame, text="Rotation Angle (°):").pack(anchor='w', pady=(5, 2))
        rot_frame = ttk.Frame(self.params_frame)
        rot_frame.pack(fill='x', pady=2)

        rot_var = tk.IntVar(value=self.params['step_2_rotate_flip']['rotate'])
        rot_scale = ttk.Scale(rot_frame, from_=-180, to=180, orient='horizontal',
                              variable=rot_var,
                              command=lambda v: self.update_step2_rotate(rot_var))
        rot_scale.pack(side='left', fill='x', expand=True, padx=(0, 5))
        rot_spinbox = ttk.Spinbox(rot_frame, from_=-180, to=180, textvariable=rot_var,
                                  width=6, command=lambda: self.update_step2_rotate(rot_var))
        rot_spinbox.pack(side='right')
        rot_spinbox.bind('<Return>', lambda e: self.update_step2_rotate(rot_var))
        rot_spinbox.bind('<FocusOut>', lambda e: self.update_step2_rotate(rot_var))

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        # Flip horizontal
        flip_h_var = tk.BooleanVar(value=self.params['step_2_rotate_flip']['flip_h'])
        ttk.Checkbutton(self.params_frame, text="Flip Horizontal (left ↔ right)",
                        variable=flip_h_var,
                        command=lambda: self.update_step2_flip(flip_h_var, None)).pack(anchor='w', pady=3)

        # Flip vertical
        flip_v_var = tk.BooleanVar(value=self.params['step_2_rotate_flip']['flip_v'])
        ttk.Checkbutton(self.params_frame, text="Flip Vertical (top ↔ bottom)",
                        variable=flip_v_var,
                        command=lambda: self.update_step2_flip(None, flip_v_var)).pack(anchor='w', pady=3)

        self.param_widgets = {'rot_var': rot_var, 'flip_h_var': flip_h_var, 'flip_v_var': flip_v_var}

    def create_step3_controls(self):
        """Create controls for Step 3: Grayscale Conversion"""
        skip_var = tk.BooleanVar(value=not self.params['step_3_grayscale']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_3_grayscale', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Skipping shows RGB but processes grayscale internally.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step4_controls(self):
        """Create controls for Step 4: Sobel Gradient"""
        skip_var = tk.BooleanVar(value=not self.params['step_4_sobel']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_4_sobel', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Sobel computes image gradient as elevation map for watershed. High gradients form barriers between regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step5_controls(self):
        """Create controls for Step 5: Histogram Markers"""
        skip_var = tk.BooleanVar(value=not self.params['step_5_markers']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_5_markers', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)

        # Low threshold
        ttk.Label(self.params_frame, text="Low Threshold (Background):").pack(anchor='w', pady=(5, 2))
        low_frame = ttk.Frame(self.params_frame)
        low_frame.pack(fill='x', pady=2)
        low_var = tk.IntVar(value=self.params['step_5_markers']['low_threshold'])
        ttk.Scale(low_frame, from_=0, to=255, orient='horizontal', variable=low_var,
                  command=lambda v: self.update_step5_params(low_var, None)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        low_spinbox = ttk.Spinbox(low_frame, from_=0, to=255, textvariable=low_var,
                                  width=6, command=lambda: self.update_step5_params(low_var, None))
        low_spinbox.pack(side='right')
        low_spinbox.bind('<Return>', lambda e: self.update_step5_params(low_var, None))
        low_spinbox.bind('<FocusOut>', lambda e: self.update_step5_params(low_var, None))

        # High threshold
        ttk.Label(self.params_frame, text="High Threshold (Foreground):").pack(anchor='w', pady=(10, 2))
        high_frame = ttk.Frame(self.params_frame)
        high_frame.pack(fill='x', pady=2)
        high_var = tk.IntVar(value=self.params['step_5_markers']['high_threshold'])
        ttk.Scale(high_frame, from_=0, to=255, orient='horizontal', variable=high_var,
                  command=lambda v: self.update_step5_params(None, high_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        high_spinbox = ttk.Spinbox(high_frame, from_=0, to=255, textvariable=high_var,
                                   width=6, command=lambda: self.update_step5_params(None, high_var))
        high_spinbox.pack(side='right')
        high_spinbox.bind('<Return>', lambda e: self.update_step5_params(None, high_var))
        high_spinbox.bind('<FocusOut>', lambda e: self.update_step5_params(None, high_var))
        self.param_widgets = {'low_var': low_var, 'high_var': high_var}

    def create_step6_controls(self):
        """Create controls for Step 6: Watershed"""
        skip_var = tk.BooleanVar(value=not self.params['step_6_watershed']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_6_watershed', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Watershed floods elevation map from markers to segment regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step7_controls(self):
        """Create controls for Step 7: Morphological Close"""
        skip_var = tk.BooleanVar(value=not self.params['step_7_morph_close']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_7_morph_close', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(self.params_frame, text="Number of Iterations:").pack(anchor='w', pady=(5, 2))
        iter_frame = ttk.Frame(self.params_frame)
        iter_frame.pack(fill='x', pady=2)
        iter_var = tk.IntVar(value=self.params['step_7_morph_close']['iterations'])
        ttk.Scale(iter_frame, from_=1, to=10, orient='horizontal', variable=iter_var,
                  command=lambda v: self.update_step7_iterations(iter_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        iter_spinbox = ttk.Spinbox(iter_frame, from_=1, to=10, textvariable=iter_var,
                                   width=6, command=lambda: self.update_step7_iterations(iter_var))
        iter_spinbox.pack(side='right')
        iter_spinbox.bind('<Return>', lambda e: self.update_step7_iterations(iter_var))
        iter_spinbox.bind('<FocusOut>', lambda e: self.update_step7_iterations(iter_var))
        self.param_widgets = {'iter_var': iter_var}

    def create_step8_controls(self):
        """Create controls for Step 8: Morphological Open"""
        skip_var = tk.BooleanVar(value=not self.params['step_8_morph_open']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_8_morph_open', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(self.params_frame, text="Number of Iterations:").pack(anchor='w', pady=(5, 2))
        iter_frame = ttk.Frame(self.params_frame)
        iter_frame.pack(fill='x', pady=2)
        iter_var = tk.IntVar(value=self.params['step_8_morph_open']['iterations'])
        ttk.Scale(iter_frame, from_=1, to=10, orient='horizontal', variable=iter_var,
                  command=lambda v: self.update_step8_iterations(iter_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        iter_spinbox = ttk.Spinbox(iter_frame, from_=1, to=10, textvariable=iter_var,
                                   width=6, command=lambda: self.update_step8_iterations(iter_var))
        iter_spinbox.pack(side='right')
        iter_spinbox.bind('<Return>', lambda e: self.update_step8_iterations(iter_var))
        iter_spinbox.bind('<FocusOut>', lambda e: self.update_step8_iterations(iter_var))
        self.param_widgets = {'iter_var': iter_var}

    def create_step9_controls(self):
        """Create controls for Step 9: Fill Holes"""
        skip_var = tk.BooleanVar(value=not self.params['step_9_fillholes']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_9_fillholes', skip_var)).pack(anchor='w', pady=5)

        ttk.Label(self.params_frame, text="Note: Fills small holes inside detected regions.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)

    def create_step10_controls(self):
        """Create controls for Step 10: Labeling"""
        skip_var = tk.BooleanVar(value=not self.params['step_10_labeling']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                       variable=skip_var,
                       command=lambda: self.toggle_step_skip('step_10_labeling', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(self.params_frame, text="Minimum Size (pixels):").pack(anchor='w', pady=(5, 2))
        size_frame = ttk.Frame(self.params_frame)
        size_frame.pack(fill='x', pady=2)
        size_var = tk.IntVar(value=self.params['step_10_labeling']['min_size'])
        ttk.Scale(size_frame, from_=0, to=200, orient='horizontal', variable=size_var,
                  command=lambda v: self.update_step10_size(size_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        size_spinbox = ttk.Spinbox(size_frame, from_=0, to=200, textvariable=size_var,
                                   width=6, command=lambda: self.update_step10_size(size_var))
        size_spinbox.pack(side='right')
        size_spinbox.bind('<Return>', lambda e: self.update_step10_size(size_var))
        size_spinbox.bind('<FocusOut>', lambda e: self.update_step10_size(size_var))
        self.param_widgets = {'size_var': size_var}

    def create_step11_controls(self):
        """Create controls for Step 11: Final Result"""
        ttk.Label(self.params_frame, text="Minimum Area (pixels):").pack(anchor='w', pady=(5, 2))
        min_area_frame = ttk.Frame(self.params_frame)
        min_area_frame.pack(fill='x', pady=2)
        min_area_var = tk.IntVar(value=self.params['step_11_final']['min_area'])
        ttk.Scale(min_area_frame, from_=0, to=1500, orient='horizontal', variable=min_area_var,
                  command=lambda v: self.update_step11_min_area(min_area_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        min_spinbox = ttk.Spinbox(min_area_frame, from_=0, to=1500, textvariable=min_area_var,
                                  width=6, command=lambda: self.update_step11_min_area(min_area_var))
        min_spinbox.pack(side='right')
        min_spinbox.bind('<Return>', lambda e: self.update_step11_min_area(min_area_var))
        min_spinbox.bind('<FocusOut>', lambda e: self.update_step11_min_area(min_area_var))

        ttk.Label(self.params_frame, text="Maximum Area (pixels):").pack(anchor='w', pady=(10, 2))
        max_area_frame = ttk.Frame(self.params_frame)
        max_area_frame.pack(fill='x', pady=2)
        max_area_var = tk.IntVar(value=self.params['step_11_final']['max_area'])
        ttk.Scale(max_area_frame, from_=1000, to=50000, orient='horizontal', variable=max_area_var,
                  command=lambda v: self.update_step11_max_area(max_area_var)).pack(side='left', fill='x', expand=True, padx=(0, 5))
        max_spinbox = ttk.Spinbox(max_area_frame, from_=1000, to=50000, textvariable=max_area_var,
                                  width=6, command=lambda: self.update_step11_max_area(max_area_var))
        max_spinbox.pack(side='right')
        max_spinbox.bind('<Return>', lambda e: self.update_step11_max_area(max_area_var))
        max_spinbox.bind('<FocusOut>', lambda e: self.update_step11_max_area(max_area_var))
        ttk.Label(self.params_frame, text="Note: Segments are colored red.",
                 wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=10)
        self.param_widgets = {'min_area_var': min_area_var, 'max_area_var': max_area_var}

    def create_step12_controls(self):
        """Create controls for Step 12: Perspective Warp"""
        skip_var = tk.BooleanVar(value=not self.params['step_12_perspective']['enabled'])
        ttk.Checkbutton(self.params_frame, text="Skip this step",
                        variable=skip_var,
                        command=lambda: self.toggle_step_skip('step_12_perspective', skip_var)).pack(anchor='w', pady=5)

        ttk.Separator(self.params_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(self.params_frame,
                  text="Perspective warp rectifies the ROI polygon to a top-down view.\n\n"
                       "The 4 extreme corners of your ROI polygon (top-left, top-right, "
                       "bottom-right, bottom-left) are used as the source quad and mapped "
                       "to a fixed 272 × 272 px output (representing the 272 mm × 272 mm "
                       "panel at 10 mm height).",
                  wraplength=280, font=('Arial', 8)).pack(anchor='w', pady=5)

    # Parameter update methods
    def toggle_step_skip(self, step_key, skip_var):
        """Toggle skip for any step"""
        self.params[step_key]['enabled'] = not skip_var.get()
        self.process_all_steps()
        self.display_current_step()

    def update_step2_rotate(self, rot_var):
        """Update step 2 rotation angle"""
        try:
            value = int(rot_var.get())
            value = max(-180, min(180, value))
            rot_var.set(value)
            self.params['step_2_rotate_flip']['rotate'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            rot_var.set(self.params['step_2_rotate_flip']['rotate'])

    def update_step2_flip(self, flip_h_var, flip_v_var):
        """Update step 2 flip flags"""
        if flip_h_var is not None:
            self.params['step_2_rotate_flip']['flip_h'] = flip_h_var.get()
        if flip_v_var is not None:
            self.params['step_2_rotate_flip']['flip_v'] = flip_v_var.get()
        self.process_all_steps()
        self.display_current_step()

    def update_step5_params(self, low_var, high_var):
        """Update step 5 threshold parameters"""
        try:
            if low_var:
                value = max(0, min(255, int(low_var.get())))
                low_var.set(value)
                self.params['step_5_markers']['low_threshold'] = value
            if high_var:
                value = max(0, min(255, int(high_var.get())))
                high_var.set(value)
                self.params['step_5_markers']['high_threshold'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            if low_var:
                low_var.set(self.params['step_5_markers']['low_threshold'])
            if high_var:
                high_var.set(self.params['step_5_markers']['high_threshold'])

    def update_step7_iterations(self, iter_var):
        """Update step 7 morphological close iterations"""
        try:
            value = max(1, min(10, int(iter_var.get())))
            iter_var.set(value)
            self.params['step_7_morph_close']['iterations'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            iter_var.set(self.params['step_7_morph_close']['iterations'])

    def update_step8_iterations(self, iter_var):
        """Update step 8 morphological open iterations"""
        try:
            value = max(1, min(10, int(iter_var.get())))
            iter_var.set(value)
            self.params['step_8_morph_open']['iterations'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            iter_var.set(self.params['step_8_morph_open']['iterations'])

    def update_step10_size(self, size_var):
        """Update step 10 minimum size"""
        try:
            value = max(0, min(200, int(size_var.get())))
            size_var.set(value)
            self.params['step_10_labeling']['min_size'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            size_var.set(self.params['step_10_labeling']['min_size'])

    def update_step11_min_area(self, area_var):
        """Update step 11 minimum area"""
        try:
            value = max(0, min(1500, int(area_var.get())))
            area_var.set(value)
            self.params['step_11_final']['min_area'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            area_var.set(self.params['step_11_final']['min_area'])

    def update_step11_max_area(self, area_var):
        """Update step 11 maximum area"""
        try:
            value = max(1000, min(50000, int(area_var.get())))
            area_var.set(value)
            self.params['step_11_final']['max_area'] = value
            self.process_all_steps()
            self.display_current_step()
        except ValueError:
            area_var.set(self.params['step_11_final']['max_area'])

    def tab2_previous_step(self):
        """Navigate to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            # Going back to Step 1 re-enters drawing mode (keep existing points)
            self.display_current_step()

    def tab2_next_step(self):
        """Navigate to next step"""
        # Moving from Step 1 → Step 2: ROI must be drawn and closed first
        if self.current_step == 1:
            if not self.step1_polygon_closed or len(self.step1_polygon_points) < 3:
                messagebox.showwarning("Warning",
                    "Please complete the ROI polygon first.\n"
                    "Click points on the image, then click near the first point to close.")
                return
            # Run full pipeline (steps 2-12) now that ROI is confirmed
            self.process_all_steps()
            self.current_step = 2
            self.display_current_step()
            self.tab2_info_label.config(
                text="ROI confirmed. Processing complete. Navigate through steps.")
            return

        if self.current_step < 12:
            self.current_step += 1
            self.display_current_step()

    def tab2_save_config(self):
        """Save current configuration to JSON.
           Defaults to the same folder as the auto-saved ROI JSON."""
        if self.tab2_roi_points is None:
            messagebox.showwarning("Warning",
                "No ROI defined yet. Please complete Step 1 ROI drawing first.")
            return

        # Default save directory = same as ROI auto-save (next to image)
        init_dir = getattr(self, '_roi_save_dir', None)
        if init_dir is None and self.tab2_image_path:
            init_dir = os.path.dirname(os.path.abspath(self.tab2_image_path))

        img_stem = Path(self.tab2_image_path).stem if self.tab2_image_path else "config"
        default_name = f"{img_stem}_config.json"

        save_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            initialdir=init_dir,
            initialfile=default_name,
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

            # ── Also write a copy to configurations/conf.json so Batch tab
            #    can auto-load it on next startup ──────────────────────────
            try:
                conf_dir = Path(__file__).resolve().parent / 'configurations'
                conf_dir.mkdir(exist_ok=True)
                with open(conf_dir / 'conf.json', 'w') as _cf:
                    json.dump(data, _cf, indent=4)
            except Exception:
                pass  # never block the main save if this fails

            messagebox.showinfo("Success", f"Configuration saved to {save_path}")

    def tab2_load_config(self):
        """Load configuration from JSON.
        When polygon_points are present, pre-loads them into the Step 1 drawing
        state so that clicking 'Start Processing' will show the saved polygon
        in Step 1 rather than an empty canvas.
        """
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load ROI points into BOTH the roi_points array AND the Step 1
            # polygon drawing state so Start Processing can restore them.
            if 'polygon_points' in data:
                pts = data['polygon_points']
                self.tab2_roi_points       = np.array(pts, dtype=np.int32)
                # Pre-fill Step 1 drawing state
                self.step1_polygon_points  = [tuple(p) for p in pts]
                self.step1_polygon_closed  = True
            else:
                # Config has no polygon — clear any previously loaded ROI
                self.tab2_roi_points      = None
                self.step1_polygon_points = []
                self.step1_polygon_closed = False

            # Load pipeline parameters
            if 'pipeline_params' in data:
                self.params = data['pipeline_params']

            messagebox.showinfo("Success", "Configuration loaded successfully")
            self.tab2_info_label.config(
                text=f"Configuration loaded "
                     f"({'ROI ready – click Start Processing to review in Step 1' if self.tab2_roi_points is not None else 'No ROI in file – draw ROI in Step 1'}).")

    # ==================== TAB 3: BATCH PROCESSING ====================

    def init_tab3(self):
        """Initialize Batch Processing tab with new GUI layout."""
        self.tab3_input_folder = None
        self.tab3_config_data = None
        self.tab3_roi_points = None
        self.tab3_processing = False

        frame = self.tab3
        frame.configure(bg=self.BG)

        def _row_entry(parent, label_text, button_text, cmd):
            """Helper: label + path entry + browse button row."""
            row = tk.Frame(parent, bg=self.PANEL)
            row.pack(fill='x', padx=16, pady=6)
            tk.Label(row, text=label_text, bg=self.PANEL, fg=self.FG2,
                     font=('Segoe UI', 9), width=14, anchor='w').pack(side='left')
            var = tk.StringVar(value='Not selected')
            entry = tk.Entry(row, textvariable=var, bg=self.ENTRY_BG, fg=self.FG,
                             relief='flat', font=('Segoe UI', 9), state='readonly',
                             readonlybackground=self.ENTRY_BG, bd=0)
            entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(6, 8))
            tk.Button(row, text=button_text, command=cmd,
                      bg=self.BTN_BG, fg=self.FG, relief='flat',
                      font=('Segoe UI', 9, 'bold'), padx=10,
                      activebackground=self.HIGHLIGHT, cursor='hand2').pack(side='right')
            return var

        # ── Settings card ────────────────────────────────────────────────
        card = tk.Frame(frame, bg=self.PANEL)
        card.pack(fill='x', padx=20, pady=(18, 0))

        tk.Label(card, text='Batch Processing', bg=self.PANEL, fg=self.HIGHLIGHT,
                 font=('Segoe UI', 13, 'bold')).pack(anchor='w', padx=16, pady=(14, 8))

        # Upload Images Folder row
        self._tab3_folder_var = _row_entry(card, 'Images Folder:', 'Upload Images Folder',
                                           self.tab3_select_folder)

        # Output Folder row
        self._tab3_output_var = _row_entry(card, 'Output Folder:', 'Select Output Folder',
                                           self._tab3_select_output_folder)

        # Config + ROI row
        misc_row = tk.Frame(card, bg=self.PANEL)
        misc_row.pack(fill='x', padx=16, pady=(4, 10))

        def _mini_btn(parent, text, cmd):
            return tk.Button(parent, text=text, command=cmd,
                             bg=self.BTN_BG, fg=self.FG, relief='flat',
                             font=('Segoe UI', 9, 'bold'), padx=10, pady=4,
                             activebackground=self.HIGHLIGHT, cursor='hand2')



        sep = tk.Frame(card, bg=self.ACCENT, height=1)
        sep.pack(fill='x', padx=16, pady=(4, 10))

        # Start Process button
        btn_row = tk.Frame(card, bg=self.PANEL)
        btn_row.pack(fill='x', padx=16, pady=(0, 14))

        self.tab3_process_button = tk.Button(btn_row, text='▶  Start Process',
                                              command=self.tab3_start_batch_processing,
                                              state='disabled',
                                              bg=self.HIGHLIGHT, fg='#ffffff', relief='flat',
                                              font=('Segoe UI', 11, 'bold'), padx=20, pady=7,
                                              activebackground=self.BTN_ACT, cursor='hand2',
                                              disabledforeground='#888888')
        self.tab3_process_button.pack(side='left')

        # ── Status + log area ────────────────────────────────────────────
        status_card = tk.Frame(frame, bg=self.PANEL)
        status_card.pack(fill='both', expand=True, padx=20, pady=12)

        status_row = tk.Frame(status_card, bg=self.PANEL)
        status_row.pack(fill='x', padx=16, pady=(12, 6))

        self.tab3_status_label = tk.Label(status_row, text='Ready',
                                          bg=self.PANEL, fg=self.HIGHLIGHT,
                                          font=('Segoe UI', 10, 'bold'))
        self.tab3_status_label.pack(side='left')

        self.tab3_progress = ttk.Progressbar(status_row, orient='horizontal',
                                              mode='determinate', length=400)
        self.tab3_progress.pack(side='right', fill='x', expand=True, padx=(16, 0))

        # Log area
        log_outer = tk.Frame(status_card, bg=self.BG)
        log_outer.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        log_sb = ttk.Scrollbar(log_outer)
        log_sb.pack(side='right', fill='y')

        self.tab3_log_text = scrolledtext.ScrolledText(
            log_outer, height=20,
            bg='#181d22', fg='#b0bec9',
            font=('Consolas', 9), relief='flat', bd=0,
            insertbackground=self.FG,
            yscrollcommand=log_sb.set)
        self.tab3_log_text.pack(fill='both', expand=True)
        log_sb.config(command=self.tab3_log_text.yview)

        # ── Auto-load configurations/conf.json (do last) ─────────────────
        # This file is written automatically every time the user saves a
        # config or closes an ROI polygon in the Single Image tab.
        _conf_path = Path(__file__).resolve().parent / 'configurations' / 'conf.json'
        if _conf_path.exists():
            try:
                with open(_conf_path, 'r') as _f:
                    _conf = json.load(_f)
                if 'polygon_points' in _conf or 'pipeline_params' in _conf:
                    self.tab3_config_data = _conf
                    if 'polygon_points' in _conf:
                        self.tab3_roi_points = np.array(_conf['polygon_points'], dtype=np.int32)
                    self.tab3_log(f"Auto-loaded configuration: {_conf_path}")
                    self.tab3_check_ready()
                else:
                    self.tab3_log("conf.json found but missing required keys (polygon_points / pipeline_params).")
            except Exception as _e:
                self.tab3_log(f"Could not load conf.json: {_e}")

    def _tab3_select_output_folder(self):
        """Select output folder."""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self._tab3_output_var.set(folder_path)

    def tab3_select_folder(self):
        """Select input folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Input Folder")

        if folder_path:
            self.tab3_input_folder = folder_path
            self._tab3_folder_var.set(folder_path)
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
        
                    self.tab3_log(f"ROI coordinates loaded: {file_path} ({len(self.tab3_roi_points)} points)")
                    self.tab3_check_ready()
                else:
                    messagebox.showerror("Error", "Invalid file. Must contain 'polygon_points'.")
                    self.tab3_roi_points = None
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROI: {str(e)}")
                self.tab3_roi_points = None

    def _normalize_params(self, params):
        """
        FIX: Translate old-style config key names (saved by the original app)
        to the new key names used by the updated pipeline.

        Original key  →  New key
        ─────────────────────────
        step_3_sobel       → step_4_sobel
        step_4_markers     → step_5_markers
        step_5_watershed   → step_6_watershed
        step_6_morph_close → step_7_morph_close
        step_7_morph_open  → step_8_morph_open
        step_8_fillholes   → step_9_fillholes
        step_9_labeling    → step_10_labeling
        step_10_final      → step_11_final

        Also adds missing 'step_2_rotate_flip' (disabled) and
        'step_12_perspective' (disabled by default) so old configs
        don't accidentally trigger the perspective warp.
        """
        KEY_MAP = {
            'step_3_sobel':       'step_4_sobel',
            'step_4_markers':     'step_5_markers',
            'step_5_watershed':   'step_6_watershed',
            'step_6_morph_close': 'step_7_morph_close',
            'step_7_morph_open':  'step_8_morph_open',
            'step_8_fillholes':   'step_9_fillholes',
            'step_9_labeling':    'step_10_labeling',
            'step_10_final':      'step_11_final',
        }
        normalized = {}
        for k, v in params.items():
            new_key = KEY_MAP.get(k, k)   # rename if in map, else keep as-is
            normalized[new_key] = v

        # Ensure rotate/flip step exists but stays OFF for old configs
        if 'step_2_rotate_flip' not in normalized:
            normalized['step_2_rotate_flip'] = {
                'enabled': False, 'rotate': 0, 'flip_h': False, 'flip_v': False
            }

        # IMPORTANT: ensure perspective warp defaults to DISABLED for old configs.
        # The default in params.get('step_12_perspective', {}).get('enabled', True)
        # would silently warp every image if this key is missing.
        if 'step_12_perspective' not in normalized:
            normalized['step_12_perspective'] = {'enabled': False}

        return normalized

    def tab3_check_ready(self):
        """Check if ready to start batch processing.
        Requires: input folder AND a config with polygon_points + pipeline_params.
        """
        has_folder = bool(self.tab3_input_folder)
        has_roi    = (self.tab3_roi_points is not None or
                      (self.tab3_config_data is not None and
                       'polygon_points' in self.tab3_config_data))
        has_params = (self.tab3_config_data is not None and
                      'pipeline_params' in self.tab3_config_data)

        if has_folder and has_roi and has_params:
            self.tab3_process_button.config(state='normal')
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

        # Get list of image files — top-level folder only (no recursion)
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []

        for file in os.listdir(self.tab3_input_folder):
            file_path = os.path.join(self.tab3_input_folder, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

        if not image_files:
            messagebox.showwarning("Warning", "No image files found in selected folder")
            self.tab3_processing = False
            self.tab3_process_button.config(state='normal')
            return

        self.tab3_log(f"\nFound {len(image_files)} image(s) to process")

        # Create results folder structure inside the selected output folder
        output_base = self._tab3_output_var.get()
        if not output_base or output_base == 'Not selected':
            output_base = self.tab3_input_folder
        defects_path     = os.path.join(output_base, "defects")
        non_defects_path = os.path.join(output_base, "non_defects")

        os.makedirs(defects_path, exist_ok=True)
        os.makedirs(non_defects_path, exist_ok=True)

        self.tab3_log(f"Created output folders:")
        self.tab3_log(f"  - {defects_path}")
        self.tab3_log(f"  - {non_defects_path}")

        # Process each image
        total = len(image_files)
        defect_count = 0
        non_defect_count = 0

        for idx, image_path in enumerate(image_files):
            # Update progress
            progress = (idx / total) * 100
            self.tab3_progress['value'] = progress
            self.tab3_status_label.config(text=f"Processing {idx + 1}/{total}: {os.path.basename(image_path)}")
            self.root.update()

            # Process image — image_path is already the full absolute path
            try:
                num_segments = self.tab3_process_single_image(image_path,
                                                                defects_path,
                                                                non_defects_path,
                                                                image_path)

                if num_segments >= 1:
                    defect_count += 1
                    self.tab3_log(f"✓ {os.path.basename(image_path)} -> defects ({num_segments} segments)")
                else:
                    non_defect_count += 1
                    self.tab3_log(f"✓ {os.path.basename(image_path)} -> non_defects ({num_segments} segment(s))")

            except Exception as e:
                self.tab3_log(f"✗ {os.path.basename(image_path)} -> ERROR: {str(e)}")

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

        # Get ROI points - use separately loaded ROI if available
        if self.tab3_roi_points is not None:
            roi_points = self.tab3_roi_points
        elif self.tab3_config_data and 'polygon_points' in self.tab3_config_data:
            roi_points = np.array(self.tab3_config_data['polygon_points'], dtype=np.int32)
        else:
            # No ROI defined – use full image as ROI
            h, w = cv_image.shape[:2]
            roi_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)

        # Get parameters from config, or fall back to built-in defaults
        # Keys match exactly those used by the Single Image (Tab 2) pipeline
        if self.tab3_config_data and 'pipeline_params' in self.tab3_config_data:
            # FIX: Normalize key names so configs saved by the original app
            # (which used old step numbers) work correctly with the new pipeline.
            params = self._normalize_params(self.tab3_config_data['pipeline_params'])
            print(f"[DEBUG] Normalized params keys: {list(params.keys())}")
        else:
            params = {
                'step_2_rotate_flip': {'enabled': False, 'rotate': 0, 'flip_h': False, 'flip_v': False},
                'step_3_grayscale':   {'enabled': True},
                'step_4_sobel':       {'enabled': True},
                'step_5_markers':     {'enabled': True, 'low_threshold': 30, 'high_threshold': 150},
                'step_6_watershed':   {'enabled': True},
                'step_7_morph_close': {'enabled': True, 'iterations': 2},
                'step_8_morph_open':  {'enabled': True, 'iterations': 2},
                'step_9_fillholes':   {'enabled': True},
                'step_10_labeling':   {'enabled': True, 'min_size': 0},
                'step_11_final':      {'min_area': 0, 'max_area': 10000},
                'step_12_perspective': {'enabled': False},
            }

        # Step 1: Apply ROI mask
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        masked_image = cv_image.copy()
        masked_image[mask == 0] = [0, 0, 0]

        # Step 2: Rotate / Flip
        rotated_bgr = masked_image.copy()
        if params.get('step_2_rotate_flip', {}).get('enabled', False):
            angle = params['step_2_rotate_flip'].get('rotate', 0)
            if angle != 0:
                h, w = rotated_bgr.shape[:2]
                M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
                rotated_bgr = cv2.warpAffine(rotated_bgr, M_rot, (w, h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
            if params['step_2_rotate_flip'].get('flip_h', False):
                rotated_bgr = cv2.flip(rotated_bgr, 1)
            if params['step_2_rotate_flip'].get('flip_v', False):
                rotated_bgr = cv2.flip(rotated_bgr, 0)

        # Step 3: Grayscale
        gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)

        # Step 4: Sobel gradient
        if params.get('step_4_sobel', {}).get('enabled', True):
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            elevation_map = np.hypot(sobel_x, sobel_y)
            elevation_display = cv2.normalize(elevation_map, None, 0, 255,
                                              cv2.NORM_MINMAX).astype(np.uint8)
        else:
            elevation_display = gray

        # Step 5: Histogram markers
        # FIX: Use _normalize_params-safe lookup. Also fall back to old key name
        # 'step_4_markers' in case a config bypassed normalization.
        _marker_cfg = params.get('step_5_markers', params.get('step_4_markers', {}))
        high_thresh = _marker_cfg.get('high_threshold', 150)  # FIX: resolve BEFORE if-block
        low_thresh  = _marker_cfg.get('low_threshold', 30)
        if _marker_cfg.get('enabled', True):
            # low_thresh and high_thresh already extracted above
            inv = 255 - gray
            background = cv2.GaussianBlur(inv, (0, 0), sigmaX=50, sigmaY=50)
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

        # Step 6: Watershed
        if params.get('step_6_watershed', {}).get('enabled', True):
            elevation_map_color = cv2.cvtColor(elevation_display, cv2.COLOR_GRAY2BGR)
            markers_watershed = markers.copy()
            segmentation = cv2.watershed(elevation_map_color, markers_watershed)
        else:
            segmentation = markers.copy()

        # Convert segmentation to binary
        binary_seg = (segmentation > high_thresh).astype(np.uint8)

        # Step 7: Morphological Close
        if params.get('step_7_morph_close', {}).get('enabled', True):
            iterations = params['step_7_morph_close'].get('iterations', 2)
            kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(binary_seg, cv2.MORPH_CLOSE, kernel,
                                           iterations=iterations)
        else:
            morph_close = binary_seg.copy()

        # Step 8: Morphological Open
        if params.get('step_8_morph_open', {}).get('enabled', True):
            iterations = params['step_8_morph_open'].get('iterations', 2)
            kernel = np.ones((3, 3), np.uint8)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel,
                                          iterations=iterations)
        else:
            morph_open = morph_close.copy()

        # Step 9: Fill holes
        if params.get('step_9_fillholes', {}).get('enabled', True):
            filled = ndi.binary_fill_holes(morph_open).astype(np.uint8) * 255
        else:
            filled = morph_open * 255

        # Step 10: Label components
        if params.get('step_10_labeling', {}).get('enabled', True):
            labeled_array, num_features = ndi.label(filled)
            min_size = params['step_10_labeling'].get('min_size', 0)
            sizes = np.bincount(labeled_array.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0
            cleaned_labels = mask_sizes[labeled_array]
            labeled_cleaned, num_cleaned = ndi.label(cleaned_labels)
        else:
            labeled_cleaned, num_cleaned = ndi.label(filled)

        # Step 11: Color valid segments on original image
        result = cv_image.copy()
        segment_color = (0, 0, 255)  # Red in BGR
        min_area = params.get('step_11_final', {}).get('min_area', 0)
        max_area = params.get('step_11_final', {}).get('max_area', 10000)

        valid_segments = 0
        for region_label in range(1, num_cleaned + 1):
            region_mask = (labeled_cleaned == region_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area <= area <= max_area:
                        valid_segments += 1
                        cv2.drawContours(result, [contour], -1, segment_color, -1)

        # Save to appropriate folder
        filename = os.path.basename(filename)

        # Step 12: Perspective Warp — apply before saving
        # FIX: Default is False. Old configs don't have this key, and we must NOT
        # silently warp every image. The _normalize_params() call above also injects
        # {'enabled': False} when the key is missing, but we double-guard here.
        warp_enabled = params.get('step_12_perspective', {}).get('enabled', False)
        if warp_enabled and roi_points is not None and len(roi_points) >= 4:
            try:
                roi_pts = roi_points.reshape(-1, 2).astype(np.float32)
                s = roi_pts.sum(axis=1)
                d = np.diff(roi_pts, axis=1).ravel()
                tl = roi_pts[np.argmin(s)]
                br = roi_pts[np.argmax(s)]
                tr = roi_pts[np.argmin(d)]
                bl = roi_pts[np.argmax(d)]
                src_quad = np.array([tl, tr, br, bl], dtype=np.float32)
                out_w, out_h = 272, 272
                dst_quad = np.array([[0, 0], [out_w - 1, 0],
                                     [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_quad, dst_quad)
                result = cv2.warpPerspective(result, M, (out_w, out_h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
            except Exception:
                pass  # if warp fails, fall back to un-warped result

        if valid_segments >= 1:
            output_path = os.path.join(defects_folder, filename)
        else:
            output_path = os.path.join(non_defects_folder, filename)

        cv2.imwrite(output_path, result)
        return valid_segments

class LoginWindow:
    """Login screen shown before the main app."""

    USERNAME = 'admin'
    PASSWORD = 'admin'

    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection – Login")
        self.root.geometry("420x520")
        self.root.resizable(False, False)

        BG      = '#1e2328'
        PANEL   = '#252b33'
        ACCENT  = '#2e3a4a'
        BLUE    = '#2d7dd2'
        FG      = '#dde3ea'
        FG2     = '#7f8c9a'
        ENTRY   = '#181d22'

        self.root.configure(bg=BG)

        # ── Card ──────────────────────────────────────────────────────
        card = tk.Frame(self.root, bg=PANEL, bd=0)
        card.place(relx=0.5, rely=0.5, anchor='center', width=340, height=480)

        # Logo / title
        tk.Label(card, text='⬡', bg=PANEL, fg=BLUE,
                 font=('Segoe UI', 36)).pack(pady=(36, 0))
        tk.Label(card, text='DEFECT DETECTION', bg=PANEL, fg=FG,
                 font=('Segoe UI', 13, 'bold')).pack()
        tk.Label(card, text='ABC', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9)).pack(pady=(2, 28))

        # Username
        tk.Label(card, text='Username', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9), anchor='w').pack(fill='x', padx=36)
        self.user_var = tk.StringVar()
        user_entry = tk.Entry(card, textvariable=self.user_var,
                              bg=ENTRY, fg=FG, relief='flat',
                              font=('Segoe UI', 11), insertbackground=FG, bd=0)
        user_entry.pack(fill='x', padx=36, ipady=8, pady=(3, 14))

        # Password
        tk.Label(card, text='Password', bg=PANEL, fg=FG2,
                 font=('Segoe UI', 9), anchor='w').pack(fill='x', padx=36)
        self.pass_var = tk.StringVar()
        self.pass_entry = tk.Entry(card, textvariable=self.pass_var,
                                   show='●', bg=ENTRY, fg=FG, relief='flat',
                                   font=('Segoe UI', 11), insertbackground=FG, bd=0)
        self.pass_entry.pack(fill='x', padx=36, ipady=8, pady=(3, 6))

        # Error label
        self.err_label = tk.Label(card, text='', bg=PANEL, fg='#e05c5c',
                                  font=('Segoe UI', 9))
        self.err_label.pack(pady=(0, 12))

        # Login button
        login_btn = tk.Button(card, text='Login',
                              command=self._try_login,
                              bg=BLUE, fg='#ffffff', relief='flat',
                              font=('Segoe UI', 11, 'bold'), padx=20, pady=8,
                              activebackground='#1a5fa8', activeforeground='#ffffff',
                              cursor='hand2')
        login_btn.pack(padx=36, fill='x')

        # Skip / continue as guest
        skip_btn = tk.Button(card, text='Continue without login \u2192',
                             command=self._skip_login,
                             bg=PANEL, fg=FG2, relief='flat',
                             font=('Segoe UI', 9), pady=4,
                             activebackground=PANEL, activeforeground=FG,
                             cursor='hand2', bd=0)
        skip_btn.pack(pady=(8, 0))

        # ── Bindings ─────────────────────────────────────────────────
        user_entry.bind('<Return>', lambda e: self.pass_entry.focus())
        self.pass_entry.bind('<Return>', lambda e: self._try_login())

        user_entry.focus()

    def _try_login(self):
        u = self.user_var.get().strip()
        p = self.pass_var.get().strip()
        if u == self.USERNAME and p == self.PASSWORD:
            self.root.destroy()
            _launch_main()
        else:
            self.err_label.config(text='Invalid username or password.')
            self.pass_var.set('')

    def _skip_login(self):
        self.root.destroy()
        _launch_main()


def _launch_main():
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()


def _launch_login():
    root = tk.Tk()
    LoginWindow(root)
    root.mainloop()


class SplashScreen:
    """Animated splash / intro screen shown before the login window."""

    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection System")
        self.root.geometry("860x540")
        self.root.resizable(False, False)
        self.root.configure(bg='#0d1117')
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - 860) // 2
        y  = (sh - 540) // 2
        self.root.geometry(f"860x540+{x}+{y}")

        BG     = '#0d1117'
        PANEL  = '#161b22'
        BLUE   = '#2d7dd2'
        BLUE2  = '#1a5fa8'
        CYAN   = '#58c4dd'
        FG     = '#e6edf3'
        FG2    = '#8b949e'
        ACCENT = '#21262d'

        self.canvas = tk.Canvas(self.root, width=860, height=540,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Background grid
        for gx in range(0, 860, 40):
            self.canvas.create_line(gx, 0, gx, 540, fill='#161b22', width=1)
        for gy in range(0, 540, 40):
            self.canvas.create_line(0, gy, 860, gy, fill='#161b22', width=1)

        # Left accent bar
        self.canvas.create_rectangle(0, 0, 5, 540, fill=BLUE, outline='')

        # Org badge top-left
        self.canvas.create_text(28, 22, text='ABC  \u00b7  ABC',
                                fill=FG2, font=('Segoe UI', 9, 'bold'), anchor='w')
        self.canvas.create_line(28, 34, 220, 34, fill=ACCENT, width=1)

        # Bottom status bar
        self.canvas.create_rectangle(0, 510, 860, 540, fill=PANEL, outline='')
        self.canvas.create_line(0, 510, 860, 510, fill=BLUE2, width=1)
        self._status_id = self.canvas.create_text(
            28, 525, text='Initialising system\u2026',
            fill=FG2, font=('Consolas', 9), anchor='w')
        self.canvas.create_text(
            832, 525, text='v3.0  |  BUILD 2025',
            fill=FG2, font=('Consolas', 9), anchor='e')

        # Hexagon logo
        cx, cy, r = 170, 265, 68
        import math as _math
        pts_outer = []
        for i in range(6):
            a = _math.radians(60 * i - 30)
            pts_outer += [cx + r * _math.cos(a), cy + r * _math.sin(a)]
        self.canvas.create_polygon(pts_outer, outline=BLUE, fill='#0d1a2a', width=2)
        pts_inner = []
        ri = 48
        for i in range(6):
            a = _math.radians(60 * i - 30)
            pts_inner += [cx + ri * _math.cos(a), cy + ri * _math.sin(a)]
        self.canvas.create_polygon(pts_inner, outline=CYAN, fill='#071020', width=1)
        self.canvas.create_text(cx, cy, text='\u2b21',
                                fill=BLUE, font=('Segoe UI', 42))

        # Corner brackets around hex
        br_off, br_len, br_w = 84, 18, 2
        for bx, by, xs, ys in [(cx - br_off, cy - br_off, 1, 1),
                                (cx + br_off, cy - br_off, -1, 1),
                                (cx + br_off, cy + br_off, -1, -1),
                                (cx - br_off, cy + br_off, 1, -1)]:
            self.canvas.create_line(bx, by, bx + xs * br_len, by,
                                    fill=CYAN, width=br_w)
            self.canvas.create_line(bx, by, bx, by + ys * br_len,
                                    fill=CYAN, width=br_w)

        # Title block
        self.canvas.create_text(278, 222,
                                text='DEFECT DETECTION SYSTEM',
                                fill=FG, font=('Segoe UI', 22, 'bold'), anchor='w')
        self.canvas.create_text(278, 254,
                                text='Advanced Image Analysis Pipeline',
                                fill=CYAN, font=('Segoe UI', 11), anchor='w')
        self.canvas.create_line(278, 272, 835, 272, fill=BLUE2, width=1)

        # Spec rows
        specs = [
            ('MODULE',   'Watershed Segmentation  \u00b7  Scikit-Image'),
            ('PIPELINE', '11-Step Analysis  \u00b7  ROI + Batch Processing'),
            ('OUTPUT',   'Defect Classification  \u00b7  Perspective Warp'),
            ('PLATFORM', 'Python  \u00b7  OpenCV  \u00b7  Tkinter  \u00b7  NumPy'),
        ]
        sy = 292
        for lbl, val in specs:
            self.canvas.create_text(278, sy, text=lbl,
                                    fill=BLUE, font=('Consolas', 8, 'bold'), anchor='w')
            self.canvas.create_text(368, sy, text=val,
                                    fill=FG2, font=('Consolas', 8), anchor='w')
            sy += 18

        # Classification badges
        bx_start = 278
        for badge_text, badge_col in [(' OFFICIAL ', '#1a3a5c'),
                                       (' RESTRICTED ', '#3a1a1a'),
                                       (' ABC ', '#1a3a1a')]:
            lbl = tk.Label(self.root, text=badge_text,
                           bg=badge_col, fg=FG2,
                           font=('Consolas', 7, 'bold'),
                           relief='flat', padx=4, pady=2)
            self.canvas.create_window(bx_start, 378, window=lbl, anchor='w')
            bx_start += len(badge_text) * 7 + 12

        # Progress bar
        bx1, by1, bx2, by2 = 278, 410, 835, 424
        self.canvas.create_rectangle(bx1, by1, bx2, by2,
                                     fill=ACCENT, outline=BLUE2, width=1)
        self._bar = self.canvas.create_rectangle(
            bx1 + 1, by1 + 1, bx1 + 1, by2 - 1, fill=BLUE, outline='')
        self._bar_x1    = bx1 + 1
        self._bar_width = (bx2 - 1) - (bx1 + 1)
        self._pct_id = self.canvas.create_text(
            bx1, by1 - 10, text='0%',
            fill=CYAN, font=('Consolas', 8, 'bold'), anchor='w')

        # Animation state
        self._progress = 0
        self._msg_idx  = 0
        self._step_msgs = [
            (0,   'Initialising system\u2026'),
            (12,  'Loading image processing modules\u2026'),
            (28,  'Configuring watershed pipeline\u2026'),
            (45,  'Preparing ROI segmentation engine\u2026'),
            (62,  'Loading batch processing routines\u2026'),
            (78,  'Calibrating defect classifier\u2026'),
            (90,  'Finalising perspective warp module\u2026'),
            (100, 'System ready.'),
        ]
        self._animate()

    def _animate(self):
        if self._progress >= 100:
            self._finish()
            return
        self._progress = min(self._progress + 1, 100)
        pct = self._progress
        new_right = self._bar_x1 + int(self._bar_width * pct / 100)
        coords = self.canvas.coords(self._bar)
        self.canvas.coords(self._bar, coords[0], coords[1], new_right, coords[3])
        self.canvas.itemconfig(self._pct_id, text=f'{pct}%')
        while (self._msg_idx < len(self._step_msgs) - 1 and
               pct >= self._step_msgs[self._msg_idx + 1][0]):
            self._msg_idx += 1
        self.canvas.itemconfig(self._status_id,
                               text=self._step_msgs[self._msg_idx][1])
        delay = 18 if pct < 95 else 40
        self.root.after(delay, self._animate)

    def _finish(self):
        BLUE  = '#2d7dd2'
        BLUE2 = '#1a5fa8'
        FG    = '#e6edf3'
        btn_frame = tk.Frame(self.root, bg='#161b22', bd=0)
        self.canvas.create_window(557, 460, window=btn_frame)
        tk.Button(btn_frame,
                  text='  ENTER SYSTEM  \u203a',
                  command=self._open_login,
                  bg=BLUE, fg=FG, relief='flat',
                  font=('Segoe UI', 11, 'bold'),
                  padx=28, pady=8,
                  activebackground=BLUE2,
                  activeforeground='#ffffff',
                  cursor='hand2', bd=0).pack()
        self.root.bind('<Return>', lambda e: self._open_login())

    def _open_login(self):
        self.root.destroy()
        _launch_login()


def main():
    _launch_main()


if __name__ == "__main__":
    main()


    #this code copy kar aur name rakho defects_final
