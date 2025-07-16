"""
Python Image Viewer with Problem Marking
This script provides a simple GUI application to view images in a selected folder.
Users can navigate through images, mark them as problematic, and save the filenames of problematic images to a text file.

Requirements:
- Python 3.x
- Tkinter (usually included with Python installations)
- Pillow (for image handling, install via pip: `pip install Pillow`)

Usage:
1. Run the script.
2. Click "Select Folder" to choose a directory containing images.
3. Use the left and right arrow keys to navigate through images.
4. Press the spacebar to mark the current image as problematic.
5. The application will display a message at the bottom if the current image is marked as problematic.
6. The filenames of problematic images will be saved in a file named `problems.txt` in the selected directory.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python Image Viewer")
        self.geometry("1200x800")

        # Initialize state
        self.image_dir = None
        self.image_paths = []
        self.index = 0
        self.problems = []  # List to store file paths marked as problematic

        # Top frame for controls
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        select_btn = tk.Button(control_frame, text="Select Folder", command=self.select_folder)
        select_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Label to display image
        self.label = tk.Label(self)
        self.label.pack(expand=True)

        # Label at bottom to show problem messages
        self.problem_label = tk.Label(self, text="", bg="red", fg="white", font=("Arial", 14))
        self.problem_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind navigation and action keys
        self.bind("<Left>", self.show_prev)
        self.bind("<Right>", self.show_next)
        self.bind("<Up>", self.mark_problem)
        self.bind("<space>", self.mark_problem)

    def select_folder(self):
        # Open directory chooser
        directory = filedialog.askdirectory()
        if not directory:
            return
        self.image_dir = directory
        self.problems = []
        # Load images and reset index
        self.image_paths = self._load_images(directory)
        if not self.image_paths:
            tk.messagebox.showerror("Error", f"No images found in {directory}")
            return
        self.index = 0
        self._show_image()

    def _load_images(self, directory):
        exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(exts)
        )

    def _show_image(self):
        # Display current image and its problem status
        path = self.image_paths[self.index]
        img = Image.open(path)
        w, h = img.size
        max_w, max_h = 1200, 800
        ratio = min(max_w / w, max_h / h)
        new_size = (int(w * ratio), int(h * ratio))

        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        img = img.resize(new_size, resample_filter)

        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

        # Update problem message based on saved state
        if path in self.problems:
            self.problem_label.config(text="자동차 검출 오류")
        else:
            self.problem_label.config(text="")

        # Update title
        self.title(f"{os.path.basename(path)} ({self.index+1}/{len(self.image_paths)}) - 문제 {len(self.problems)}개")

    def show_prev(self, event=None):
        if not self.image_paths:
            return
        self.index = (self.index - 1) % len(self.image_paths)
        self._show_image()

    def show_next(self, event=None):
        if not self.image_paths:
            return
        self.index = (self.index + 1) % len(self.image_paths)
        self._show_image()

    def mark_problem(self, event=None):
        if not self.image_paths:
            return
        path = self.image_paths[self.index]
        if path in self.problems:
            self.problems.remove(path)
            self.problem_label.config(text="")
            print(f"Unmarked problematic: {os.path.basename(path)}")
        else:
            self.problems.append(path)
            self.problem_label.config(text="자동차 검출 오류")
            print(f"Marked problematic: {os.path.basename(path)}")
        # Update title
        self.title(f"{os.path.basename(path)} ({self.index+1}/{len(self.image_paths)}) - 문제 {len(self.problems)}개")

        # Write to problems.txt using filenames only
        try:
            problems_file = os.path.join(self.image_dir, 'problems.txt')
            with open(problems_file, 'w', encoding='utf-8') as f:
                for p in self.problems:
                    f.write(os.path.basename(p) + os.linesep)
        except Exception as e:
            print(f"Error writing problems.txt: {e}")



if __name__ == '__main__':
    app = ImageViewer()
    app.mainloop()
