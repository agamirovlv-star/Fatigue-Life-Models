import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import os
import math
import re
import importlib
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageTk  # Добавляем для работы с изображениями

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

# Try to import numpy for calculations
try:
    import numpy as np
    import scipy.stats as stats
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlotSeries:
    x: List[float]
    y: List[float]
    type: int  # 0: points only, 1: lines only, 2: both points and lines
    legend: str
    color: str


@dataclass
class PlotData:
    series: List[PlotSeries]
    title: str
    x_label: str
    y_label: str
    plot_type: int  # 0: regular, 1: normal probability paper, 2: weibull probability paper


# ============================================================================
# Colors class
# ============================================================================

class Colors:
    PRIMARY = "#2196F3"
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    DANGER = "#F44336"
    DARK = "black"
    GRAY = "#666666"
    LIGHT_GRAY = "#E0E0E0"
    BACKGROUND = "#F5F5F5"
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    MENU_BG = "#F0F0F0"
    MENU_HOVER = "#E5F3FF"
    MENU_ACTIVE = "#CCE4FF"
    STATUS_BG = "#4A90E2"
    STATUS_TEXT = "#FFFFFF"
    
    GRAPH_COLORS = {
        "red": "#FF0000",
        "blue": "#2196F3",
        "green": "#4CAF50",
        "orange": "#FF9800",
        "purple": "#9C27B0",
        "black": "#000000",
        "cyan": "#00BCD4",
        "magenta": "#E91E63",
        "yellow": "#FFEB3B",
        "brown": "#795548",
        "pink": "#FF69B4",
        "gray": "#808080"
    }
    
    @classmethod
    def get_color(cls, color_name: str) -> str:
        return cls.GRAPH_COLORS.get(color_name.lower(), "#2196F3")


# ============================================================================
# Statistics Engine (fallback)
# ============================================================================

class StatisticsEngine:
    @staticmethod
    def run_calculation(command: str, title: str) -> str:
        results = []
        results.append("=" * 60)
        results.append(f"Statistical Analysis: {title}")
        results.append(f"Command: {command}")
        results.append("=" * 60)
        results.append("\n✓ Calculation completed")
        results.append("\n" + "=" * 60)
        return "\n".join(results)


# ============================================================================
# Menu Loader
# ============================================================================

class MenuLoader:
    @staticmethod
    def load_menu(filepath: str) -> Tuple[Dict[str, str], List]:
        commands = {}
        menu_structure = []
        
        try:
            with open(filepath, 'r', encoding='windows-1251') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return commands, []
        
        stack = [menu_structure]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('SUBMENU'):
                name = line[7:].strip()
                display_name = name.replace('&', '')
                menu_item = {
                    'type': 'submenu',
                    'name': display_name,
                    'original_name': name,
                    'children': []
                }
                stack[-1].append(menu_item)
                stack.append(menu_item['children'])
            
            elif line.startswith('MENU'):
                name = line[4:].strip()
                display_name = name.replace('&', '')
                commands[name] = name
                menu_item = {
                    'type': 'menu',
                    'name': display_name,
                    'original_name': name,
                    'command': name
                }
                stack[-1].append(menu_item)
            
            elif line.startswith('ENDMENU'):
                if len(stack) > 1:
                    stack.pop()
        
        return commands, menu_structure


# ============================================================================
# Dropdown Menu
# ============================================================================

class DropdownMenu:
    def __init__(self, parent, text, items=None, command=None):
        self.parent = parent
        self.text = text
        self.items = items or []
        self.command = command
        self.menu = None
        self.button = None
        self.is_active = False
        self.create_button()
    
    def create_button(self):
        self.button = tk.Button(
            self.parent,
            text=self.text,
            bg=Colors.MENU_BG,
            fg=Colors.DARK,
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground=Colors.MENU_BG,
            activeforeground=Colors.DARK,
            bd=0,
            highlightthickness=0
        )
        self.button.pack(side=tk.LEFT, padx=0)
        self.button.bind("<Button-1>", self.on_click)
    
    def on_click(self, e):
        if self.menu and self.menu.winfo_exists():
            self.close_menu()
            self.deactivate()
        else:
            self.parent.event_generate("<<CloseAllMenus>>")
            self.show_menu()
            self.activate()
    
    def activate(self):
        self.is_active = True
        self.button.configure(bg=Colors.MENU_ACTIVE)
    
    def deactivate(self):
        self.is_active = False
        self.button.configure(bg=Colors.MENU_BG)
    
    def show_menu(self):
        if self.menu and self.menu.winfo_exists():
            self.menu.destroy()
        
        self.menu = tk.Menu(self.parent, tearoff=0, bg=Colors.WHITE, 
                           fg=Colors.DARK, activebackground=Colors.MENU_HOVER,
                           activeforeground=Colors.PRIMARY, 
                           font=("Segoe UI", 10),
                           relief=tk.FLAT, bd=1)
        
        self.build_menu(self.items, self.menu)
        
        x = self.button.winfo_rootx()
        y = self.button.winfo_rooty() + self.button.winfo_height()
        self.menu.post(x, y)
        self.menu.bind("<FocusOut>", lambda e: self.on_menu_focus_out())
        self.menu.focus_set()
    
    def on_menu_focus_out(self):
        self.close_menu()
        self.deactivate()
    
    def close_menu(self):
        if self.menu and self.menu.winfo_exists():
            self.menu.destroy()
        self.menu = None
    
    def build_menu(self, items, menu):
        for item in items:
            if item['type'] == 'submenu':
                submenu = tk.Menu(menu, tearoff=0, bg=Colors.WHITE,
                                 fg=Colors.DARK, activebackground=Colors.MENU_HOVER,
                                 activeforeground=Colors.PRIMARY,
                                 font=("Segoe UI", 10))
                self.build_menu(item['children'], submenu)
                menu.add_cascade(label=item['name'], menu=submenu)
            elif item['type'] == 'menu':
                menu.add_command(
                    label=item['name'],
                    command=lambda cmd=item['command']: self.on_select(cmd)
                )
    
    def on_select(self, command):
        self.close_menu()
        self.deactivate()
        if self.command:
            self.command(command)


# ============================================================================
# Graph Frame with mouse coordinates
# ============================================================================

class GraphFrame(tk.Frame):
    def __init__(self, parent, status_callback=None):
        super().__init__(parent, bg='white')
        self.figure = None
        self.canvas = None
        self.axes = None
        self.status_callback = status_callback
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_graph()
        else:
            self.setup_placeholder()
    
    def setup_graph(self):
        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
    
    def setup_placeholder(self):
        label = tk.Label(self, text="Graph not available\nInstall matplotlib for graphs",
                        font=("Segoe UI", 12), bg='white', fg='gray')
        label.pack(expand=True)
    
    def on_mouse_move(self, event):
        if event.inaxes and self.status_callback:
            x = event.xdata
            y = event.ydata
            if x is not None and y is not None:
                self.status_callback(f"📍 X: {x:.4f}, Y: {y:.4f}")
    
    def draw_graph(self, plot_data: PlotData):
        if not MATPLOTLIB_AVAILABLE or not plot_data.series:
            return
        
        self.figure.clear()
        
        if plot_data.plot_type == 0:
            self._draw_regular_graph(plot_data)
        elif plot_data.plot_type == 1:
            self._draw_normal_probability_paper(plot_data)
        elif plot_data.plot_type == 2:
            self._draw_weibull_probability_paper(plot_data)
        
        self.canvas.draw()
    
    def _draw_regular_graph(self, plot_data: PlotData):
        ax = self.figure.add_subplot(111)
        
        all_x = [x for series in plot_data.series for x in series.x]
        all_y = [y for series in plot_data.series for y in series.y]
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 1
            y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Draw dense vertical grid
        x_min, x_max = ax.get_xlim()
        x_ticks = np.linspace(x_min, x_max, 15)
        for x_tick in x_ticks:
            ax.axvline(x=x_tick, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        y_min, y_max = ax.get_ylim()
        y_ticks = np.linspace(y_min, y_max, 10)
        for y_tick in y_ticks:
            ax.axhline(y=y_tick, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for series in plot_data.series:
            color = Colors.get_color(series.color)
            
            if series.type == 1:
                ax.plot(series.x, series.y, color=color, linewidth=2, 
                       label=series.legend, marker='', linestyle='-')
            elif series.type == 0:
                ax.scatter(series.x, series.y, color=color, s=40, 
                          zorder=5, marker='s', alpha=0.8)
            elif series.type == 2:
                ax.plot(series.x, series.y, color=color, linewidth=2, 
                       label=series.legend, marker='s', markersize=6, linestyle='-')
        
        ax.set_xlabel(plot_data.x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(plot_data.y_label, fontsize=11, fontweight='bold')
        ax.set_title(plot_data.title, fontsize=14, fontweight='bold', pad=20)
        
        if len(plot_data.series) > 0:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        ax.tick_params(axis='both', labelsize=10)
    
    def _draw_normal_probability_paper(self, plot_data: PlotData):
        ax = self.figure.add_subplot(111)
        
        prob_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        quantiles = [stats.norm.ppf(p) for p in prob_levels]
        
        all_x = [x for series in plot_data.series for x in series.x]
        all_y = [y for series in plot_data.series for y in series.y]
        
        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 1
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        for q, p in zip(quantiles, prob_levels):
            y_min_plot, y_max_plot = ax.get_ylim()
            if y_min_plot <= q <= y_max_plot:
                ax.axhline(y=q, color='lightgray', linestyle=':', linewidth=1, alpha=0.7)
                if p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
                    label = f'{p*100:.1f}%' if p < 0.1 else f'{p*100:.0f}%'
                    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, 
                           q, label, fontsize=8, va='center', color='gray')
        
        x_min, x_max = ax.get_xlim()
        x_ticks = np.linspace(x_min, x_max, 15)
        for x_tick in x_ticks:
            ax.axvline(x=x_tick, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for series in plot_data.series:
            color = Colors.get_color(series.color)
            
            if series.type == 1:
                ax.plot(series.x, series.y, color=color, linewidth=2, 
                       label=series.legend, marker='', linestyle='-')
            elif series.type == 0:
                ax.scatter(series.x, series.y, color=color, s=40, 
                          zorder=5, marker='s', alpha=0.8)
            elif series.type == 2:
                ax.plot(series.x, series.y, color=color, linewidth=2, 
                       label=series.legend, marker='s', markersize=6, linestyle='-')
        
        ax.set_xlabel(plot_data.x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Quantile', fontsize=11, fontweight='bold')
        ax.set_title(f'{plot_data.title} - Normal Probability Paper', 
                    fontsize=14, fontweight='bold', pad=20)
        
        if len(plot_data.series) > 0:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        ax.tick_params(axis='both', labelsize=10)
    
    def _draw_weibull_probability_paper(self, plot_data: PlotData):
        ax = self.figure.add_subplot(111)
        
        prob_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
                       0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        weibull_quantiles = [math.log(-math.log(1 - p)) for p in prob_levels]
        
        all_x = [x for series in plot_data.series for x in series.x]
        all_y = [y for series in plot_data.series for y in series.y]
        
        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 1
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        for wq, p in zip(weibull_quantiles, prob_levels):
            y_min_plot, y_max_plot = ax.get_ylim()
            if y_min_plot <= wq <= y_max_plot:
                ax.axhline(y=wq, color='lightgray', linestyle=':', linewidth=1, alpha=0.7)
                if p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
                    label = f'{p*100:.0f}%'
                    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
                           wq, label, fontsize=8, va='center', color='gray')
        
        x_min, x_max = ax.get_xlim()
        x_ticks = np.linspace(x_min, x_max, 15)
        for x_tick in x_ticks:
            ax.axvline(x=x_tick, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for series in plot_data.series:
            color = Colors.get_color(series.color)
            
            if series.type == 1:
                ax.plot(series.x, series.y, color=color, linewidth=2,
                       label=series.legend, marker='', linestyle='-')
            elif series.type == 0:
                ax.scatter(series.x, series.y, color=color, s=40,
                          zorder=5, marker='s', alpha=0.8)
            elif series.type == 2:
                ax.plot(series.x, series.y, color=color, linewidth=2,
                       label=series.legend, marker='s', markersize=6,
                       linestyle='-')
        
        ax.set_xlabel(plot_data.x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('ln(-ln(1-F(x)))', fontsize=11, fontweight='bold')
        ax.set_title(f'{plot_data.title} - Weibull Probability Paper',
                    fontsize=14, fontweight='bold', pad=20)
        
        if len(plot_data.series) > 0:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        ax.tick_params(axis='both', labelsize=10)


# ============================================================================
# Image Display Frame (для отображения PNG)
# ============================================================================

class ImageDisplayFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='white')
        self.image_label = None
        self.current_image_path = None
        self.photo_images = []  # Список для хранения ссылок на изображения
        
        # Создаем фрейм для изображения без прокрутки
        self.image_frame = tk.Frame(self, bg='white')
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Изначально показываем подсказку
        self.show_placeholder()
    
    def show_placeholder(self):
        """Показывает заглушку, когда изображение не загружено"""
        self.clear_image()
        placeholder = tk.Label(
            self.image_frame,
            text="📊\n\nNo image loaded\nClick 'Graph' or 'Open' to load an image",
            font=("Segoe UI", 14),
            bg='white',
            fg='gray',
            justify=tk.CENTER
        )
        placeholder.pack(expand=True, fill=tk.BOTH)
        self.image_label = placeholder
    
    def display_image(self, image_path: str):
        """Отображает PNG изображение в фрейме"""
        try:
            # Открываем изображение через PIL
            pil_image = Image.open(image_path)
            
            # Получаем размеры фрейма
            frame_width = self.image_frame.winfo_width() if self.image_frame.winfo_width() > 100 else 800
            frame_height = self.image_frame.winfo_height() if self.image_frame.winfo_height() > 100 else 600
            
            # Масштабируем изображение с сохранением пропорций
            img_width, img_height = pil_image.size
            
            # Если изображение больше фрейма, масштабируем
            if img_width > frame_width or img_height > frame_height:
                ratio = min(frame_width / img_width, frame_height / img_height)
                new_width = int(img_width * ratio * 0.95)  # 5% отступ
                new_height = int(img_height * ratio * 0.95)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Конвертируем в PhotoImage для отображения в tkinter
            photo = ImageTk.PhotoImage(pil_image)
            
            # Сохраняем ссылку, чтобы изображение не было уничтожено сборщиком мусора
            self.photo_images.append(photo)
            
            # Очищаем фрейм
            self.clear_image()
            
            # Создаем метку с изображением
            self.image_label = tk.Label(
                self.image_frame,
                image=photo,
                bg='white'
            )
            self.image_label.pack(expand=True)
            
            self.current_image_path = image_path
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.show_placeholder()
            return False
    
    def clear_image(self):
        """Очищает фрейм от всех виджетов"""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.image_label = None
        self.current_image_path = None
    
    def on_frame_configure(self, event):
        """Обновляет изображение при изменении размера фрейма"""
        if self.current_image_path:
            # Перезагружаем изображение при изменении размера
            self.display_image(self.current_image_path)


# ============================================================================
# Text Display Frame with editing and save capabilities
# ============================================================================

class TextDisplayFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='white')
        self.current_filepath = None
        self.is_modified = False
        self.save_callback = None  # Callback to update main window status
        
        # Main frame with text widget
        main_frame = tk.Frame(self, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label for file info (inside text frame)
        self.file_status = tk.Label(
            main_frame,
            text="",
            bg='white',
            fg=Colors.GRAY,
            font=("Segoe UI", 9),
            anchor=tk.W
        )
        self.file_status.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        # Text widget
        text_frame = tk.Frame(main_frame, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Times New Roman", 14),
            bg='white',
            fg='#000000',
            relief=tk.FLAT,
            bd=1
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Bind modification events
        self.text_widget.bind('<<Modified>>', self.on_text_modified)
    
    def set_save_callback(self, callback):
        """Set callback to notify main window about save status"""
        self.save_callback = callback
    
    def on_text_modified(self, event=None):
        """Handle text modification"""
        if self.text_widget.edit_modified():
            self.is_modified = True
            if self.current_filepath:
                filename = os.path.basename(self.current_filepath)
                self.file_status.config(text=f"✏️ {filename} (modified)", fg=Colors.WARNING)
                # Notify main window to enable save buttons
                if self.save_callback:
                    self.save_callback(True)
            self.text_widget.edit_modified(False)
    
    def display_text(self, text: str, filepath: str = None):
        """Display text and optionally set file path"""
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, text)
        self.text_widget.edit_modified(False)
        self.is_modified = False
        
        if filepath:
            self.current_filepath = filepath
            filename = os.path.basename(filepath)
            self.file_status.config(text=f"📄 {filename}", fg=Colors.GRAY)
        else:
            self.current_filepath = None
            self.file_status.config(text="", fg=Colors.GRAY)
        
        # Notify main window about save status
        if self.save_callback:
            self.save_callback(False)
    
    def display_file(self, filepath: str):
        """Display file content and store filepath"""
        try:
            # Try different encodings
            encodings = ['windows-1251', 'utf-8', 'cp1251', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                self.display_text(f"Error: Could not read file {filepath} with any encoding", None)
                return
                
            self.display_text(content, filepath)
        except FileNotFoundError:
            self.display_text(f"File not found: {filepath}", None)
        except Exception as e:
            self.display_text(f"Error reading file: {e}", None)
    
    def save_file(self):
        """Save current content to current filepath"""
        if not self.current_filepath:
            return False
        
        try:
            content = self.text_widget.get(1.0, tk.END).rstrip()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.current_filepath), exist_ok=True)
            
            # Try to save with appropriate encoding
            encodings = ['windows-1251', 'utf-8', 'cp1251']
            saved = False
            for encoding in encodings:
                try:
                    with open(self.current_filepath, 'w', encoding=encoding) as f:
                        f.write(content)
                    saved = True
                    break
                except UnicodeEncodeError:
                    continue
            
            if not saved:
                # If all encodings fail, save with utf-8 (most universal)
                with open(self.current_filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            self.is_modified = False
            filename = os.path.basename(self.current_filepath)
            self.file_status.config(text=f"💾 {filename} (saved)", fg=Colors.SUCCESS)
            
            # Notify main window
            if self.save_callback:
                self.save_callback(False)
            
            # Reset status after 2 seconds
            self.after(2000, lambda: self.file_status.config(
                text=f"📄 {filename}", fg=Colors.GRAY
            ))
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
            return False
    
    def save_file_as(self):
        """Save current content to a new file"""
        # Get current directory from filepath or use default
        initial_dir = os.path.dirname(self.current_filepath) if self.current_filepath else "."
        initial_file = os.path.basename(self.current_filepath) if self.current_filepath else ""
        
        filepath = filedialog.asksaveasfilename(
            title="Save File As",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".out",
            filetypes=[
                ("All files", "*.*"),
                ("Output files", "*.out"),
                ("JSON files", "*.json"),
                ("Text files", "*.txt")
            ]
        )
        
        if filepath:
            self.current_filepath = filepath
            return self.save_file()
        return False
    
    def get_current_filepath(self):
        """Return current file path"""
        return self.current_filepath
    
    def get_content(self):
        """Return current text content"""
        return self.text_widget.get(1.0, tk.END).rstrip()
    
    def is_modified(self):
        """Return whether text has been modified"""
        return self.is_modified


# ============================================================================
# Main Application
# ============================================================================

class StatisticalApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Statistical Calculator")
        self.root.geometry("1200x700")
        self.root.configure(bg=Colors.BACKGROUND)
        
        self.commands = {}
        self.menu_structure = []
        self.current_command = None
        self.current_view = 'text'
        self.top_menus = []
        
        self.stats_engine = StatisticsEngine()
        
        self.status_var = tk.StringVar()
        self.status_var.set("✅ Ready - Click 'Graph' to view an image or select a menu item")
        
        self.setup_ui()
        self.load_menu()
        
        self.root.bind("<<CloseAllMenus>>", self.on_close_all_menus)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.on_save_file())
        self.root.bind('<Control-S>', lambda e: self.on_save_file())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-O>', lambda e: self.open_file())
    
    def setup_ui(self):
        # Menu bar frame
        self.menubar_frame = tk.Frame(self.root, bg=Colors.MENU_BG, height=40)
        self.menubar_frame.pack(side=tk.TOP, fill=tk.X)
        self.menubar_frame.pack_propagate(False)
        
        # File menu
        self.file_menu = DropdownMenu(
            self.menubar_frame, "File",
            items=[{'type': 'menu', 'name': 'Exit', 'command': 'exit'}],
            command=self.on_file_menu
        )
        
        # Help menu
        self.help_menu = DropdownMenu(
            self.menubar_frame, "Help",
            items=[{'type': 'menu', 'name': 'About', 'command': 'about'}],
            command=self.on_help_menu
        )
        
        # Separator
        separator = tk.Frame(self.root, bg=Colors.LIGHT_GRAY, height=1)
        separator.pack(side=tk.TOP, fill=tk.X)
        
        # Status bar at the top
        status_frame = tk.Frame(self.root, bg=Colors.STATUS_BG, height=35)
        status_frame.pack(side=tk.TOP, fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=Colors.STATUS_BG,
            fg=Colors.STATUS_TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
            padx=15,
            pady=8
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        info_label = tk.Label(
            status_frame,
            text="ℹ️",
            bg=Colors.STATUS_BG,
            fg=Colors.STATUS_TEXT,
            font=("Segoe UI", 12),
            cursor="hand2"
        )
        info_label.pack(side=tk.RIGHT, padx=10)
        info_label.bind("<Button-1>", lambda e: self.show_status_info())
        
        # Toolbar - Main buttons
        toolbar = tk.Frame(self.root, bg=Colors.PRIMARY, height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)
        
        # Left group - Run and file operations
        self.run_btn = tk.Button(toolbar, text="▶ Run", command=self.on_run,
                                 bg=Colors.SUCCESS, fg=Colors.WHITE,
                                 relief=tk.FLAT, padx=20, pady=8,
                                 font=("Segoe UI", 10, "bold"),
                                 cursor="hand2")
        self.run_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.inp_btn = tk.Button(toolbar, text="📄 Inp", command=self.on_show_inp,
                                 bg=Colors.PRIMARY, fg=Colors.WHITE,
                                 relief=tk.FLAT, padx=20, pady=8,
                                 font=("Segoe UI", 10),
                                 cursor="hand2")
        self.inp_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.out_btn = tk.Button(toolbar, text="📋 Out", command=self.on_show_out,
                                 bg=Colors.PRIMARY, fg=Colors.WHITE,
                                 relief=tk.FLAT, padx=20, pady=8,
                                 font=("Segoe UI", 10),
                                 cursor="hand2")
        self.out_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Graph button - теперь активен с самого начала (state=tk.NORMAL)
        self.graph_btn = tk.Button(toolbar, text="📊 Graph", command=self.on_show_graph,
                                   bg=Colors.PRIMARY, fg=Colors.WHITE,
                                   relief=tk.FLAT, padx=20, pady=8,
                                   font=("Segoe UI", 10),
                                   cursor="hand2",
                                   state=tk.NORMAL)  # <-- Изменено с DISABLED на NORMAL
        self.graph_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Separator
        sep = tk.Frame(toolbar, bg=Colors.WHITE, width=2, height=30)
        sep.pack(side=tk.LEFT, padx=10)
        
        # Middle group - Open button (always active)
        self.open_btn = tk.Button(toolbar, text="📂 Open", command=self.open_file,
                                  bg=Colors.PRIMARY, fg=Colors.WHITE,
                                  relief=tk.FLAT, padx=20, pady=8,
                                  font=("Segoe UI", 10, "bold"),
                                  cursor="hand2",
                                  state=tk.NORMAL)
        self.open_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Separator
        sep2 = tk.Frame(toolbar, bg=Colors.WHITE, width=2, height=30)
        sep2.pack(side=tk.LEFT, padx=10)
        
        # Right group - Save buttons
        self.save_btn = tk.Button(toolbar, text="💾 Save", command=self.on_save_file,
                                  bg=Colors.WARNING, fg=Colors.WHITE,
                                  relief=tk.FLAT, padx=20, pady=8,
                                  font=("Segoe UI", 10, "bold"),
                                  cursor="hand2",
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_as_btn = tk.Button(toolbar, text="📁 Save As", command=self.on_save_file_as,
                                     bg=Colors.PRIMARY, fg=Colors.WHITE,
                                     relief=tk.FLAT, padx=20, pady=8,
                                     font=("Segoe UI", 10),
                                     cursor="hand2",
                                     state=tk.DISABLED)
        self.save_as_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Disable all buttons initially (except Open and Graph)
        self.run_btn.config(state=tk.DISABLED)
        self.inp_btn.config(state=tk.DISABLED)
        self.out_btn.config(state=tk.DISABLED)
        # Graph button уже активен
        # Open button is already active
        
        # Main content area
        content_frame = tk.Frame(self.root, bg=Colors.WHITE)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.graph_frame = GraphFrame(content_frame, status_callback=self.update_status)
        self.text_frame = TextDisplayFrame(content_frame)
        self.image_frame = ImageDisplayFrame(content_frame)  # Новый фрейм для изображений
        
        # Set callback for save status
        self.text_frame.set_save_callback(self.on_text_modified)
        
        # Изначально показываем текстовый фрейм
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.graph_frame.pack_forget()
        self.image_frame.pack_forget()
        self.text_frame.pack(fill=tk.BOTH, expand=True)
    
    def on_text_modified(self, modified):
        """Handle text modification notification from text frame"""
        if modified:
            self.save_btn.config(state=tk.NORMAL)
            self.save_as_btn.config(state=tk.NORMAL)
            self.status_var.set("✏️ Text modified - click Save to save changes")
        else:
            self.save_btn.config(state=tk.DISABLED)
            # Save As should always be available if there's a file
            self.save_as_btn.config(state=tk.NORMAL if self.text_frame.get_current_filepath() else tk.DISABLED)
            if not modified and self.text_frame.get_current_filepath():
                filename = os.path.basename(self.text_frame.get_current_filepath())
                self.status_var.set(f"✅ File saved: {filename}")
    
    def show_status_info(self):
        messagebox.showinfo("Status Bar Info", 
                           "Status bar shows:\n"
                           "• Current program state\n"
                           "• Selected menu item\n"
                           "• Mouse coordinates on graphs\n"
                           "• Execution status\n\n"
                           "Coordinates are shown as: 📍 X: value, Y: value")
    
    def update_status(self, text):
        self.status_var.set(text)
        self.root.update()
    
    def on_close_all_menus(self, event):
        for menu in self.top_menus:
            menu.close_menu()
            menu.deactivate()
    
    def load_menu(self):
        self.commands, self.menu_structure = MenuLoader.load_menu("menu.txt")
        
        if not self.menu_structure:
            self.status_var.set("⚠️ menu.txt not found")
            return
        
        for item in self.menu_structure:
            if item['type'] == 'submenu':
                menu = DropdownMenu(
                    self.menubar_frame,
                    item['name'],
                    items=item['children'],
                    command=self.select_command
                )
                self.top_menus.append(menu)
        
        self.status_var.set(f"✅ Loaded {len(self.commands)} menu items")
    
    def select_command(self, command: str):
        self.current_command = command
        self.root.title(f"Statistical Calculator - {command}")
        self.status_var.set(f"📌 Selected: {command}")
        
        self.run_btn.config(state=tk.NORMAL)
        self.inp_btn.config(state=tk.NORMAL)
        self.out_btn.config(state=tk.NORMAL)
        # Graph button уже активен, не меняем его состояние
        # Open button is already active, no need to change
    
    def on_file_menu(self, command):
        if command == 'exit':
            self.root.quit()
    
    def on_help_menu(self, command):
        if command == 'about':
            messagebox.showinfo("About", 
                               "Statistical Calculator\n\n"
                               "Version: 2.0\n\n"
                               "Features:\n"
                               "• Parametric and non-parametric tests\n"
                               "• Normal and Weibull probability papers\n"
                               "• Professional graphing with matplotlib\n"
                               "• Mouse coordinates display in status bar\n"
                               "• Dynamic module loading by command name\n"
                               "• Editable input/output files with save functionality\n"
                               "• Open any text file for viewing/editing\n"
                               "• View PNG images in the graph panel\n\n"
                               "Keyboard shortcuts:\n"
                               "  Ctrl+S - Save file\n"
                               "  Ctrl+O - Open file\n\n"
                               "Built with Python and tkinter")
    
    def on_run(self):
        """Execute current command - dynamically load module by name"""
        if not self.current_command:
            messagebox.showwarning("Warning", "Please select a menu item first")
            return
        
        self.status_var.set(f"🔄 Running {self.current_command}...")
        self.root.update()
        
        result = ""
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "files"))

        # Try to import and run module with the same name as command
        try:
            # Convert command name to module name (exactly as in menu)
            module_name = self.current_command
            # Try to import the module
            module = importlib.import_module(module_name)
            # Look for process function (process_CommandName)
            func_name = f"process_{module_name.lower()}"
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                # Call the function with input filename
                inp_file = f"{module_name}.json"
                success = func()

                if success:
                    # Read output file
                    out_file = f"Out/{module_name}.out"
                    if os.path.exists(out_file):
                        with open(out_file, 'r', encoding='windows-1251') as f:
                            result = f.read()
                    else:
                        result = f"Calculation completed, but output file not found"
                else:
                    result = f"Error in {module_name} calculation"
            else:
                result = f"Function {func_name} not found in {module_name}.py"
                
        except ImportError as e:
            # Module not found - use fallback
            result = f"Module {self.current_command}.py not found.\n\n"
            result += self.stats_engine.run_calculation(self.current_command, self.current_command)
        except Exception as e:
            result = f"Error: {str(e)}"
            import traceback
            result += "\n\n" + traceback.format_exc()
        
        self.text_frame.display_text(result)
        self.switch_to_text()
        self.status_var.set(f"✅ Completed: {self.current_command}")
    
    def on_show_inp(self):
        """Show input file with editing capability"""
        if not self.current_command:
            messagebox.showwarning("Warning", "Please select a menu item first")
            return
        
        # Try multiple possible input file locations
        possible_paths = [
            f"Inp/{self.current_command}.json",
            f"Inp/{self.current_command}.inp",
            f"Inp/{self.current_command}.txt",
            f"{self.current_command}.json"
        ]
        
        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath:
            self.text_frame.display_file(filepath)
            self.switch_to_text()
            filename = os.path.basename(filepath)
            self.status_var.set(f"📄 Editing input file: {filename} (click Save to save changes)")
        else:
            # Create a new file with template
            template = f"# Input file for {self.current_command}\n# Created: {self.current_command}.json\n\n"
            filepath = f"Inp/{self.current_command}.json"
            os.makedirs("Inp", exist_ok=True)
            
            # Try to create a sample template based on command type
            if 'fatigue' in self.current_command:
                template += "{\n  \"stress_levels\": [100, 150, 200, 250, 300],\n  \"cycles\": [1000, 500, 200, 100, 50]\n}"
            elif 'estimation' in self.current_command:
                template += "{\n  \"sample_data\": [10.2, 12.5, 11.8, 13.1, 9.7, 12.0]\n}"
            elif 'test' in self.current_command:
                template += "{\n  \"data\": [10.2, 12.5, 11.8, 13.1, 9.7, 12.0],\n  \"alpha\": 0.05\n}"
            else:
                template += "{\n  # Add your input data here\n  \"data\": []\n}"
            
            self.text_frame.display_text(template, filepath)
            self.switch_to_text()
            self.status_var.set(f"📄 Created new input file: {filepath} (edit and click Save)")
    
    def on_show_out(self):
        """Show output file with editing capability"""
        if not self.current_command:
            messagebox.showwarning("Warning", "Please select a menu item first")
            return
        
        # Try multiple possible output file locations
        possible_paths = [
            f"Out/{self.current_command}.out",
            f"Out/{self.current_command}.txt",
            f"{self.current_command}.out"
        ]
        
        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath:
            self.text_frame.display_file(filepath)
            self.switch_to_text()
            filename = os.path.basename(filepath)
            self.status_var.set(f"📄 Showing output file: {filename} (editable)")
        else:
            self.text_frame.display_text(f"Output file not found for: {self.current_command}\n\n"
                                          f"Searched in:\n" + "\n".join(possible_paths), None)
            self.switch_to_text()
            self.status_var.set(f"❌ Output file not found: {self.current_command}.out")
    
    def on_show_graph(self):
        """Open file dialog to select a PNG image and display it"""
        # Открываем диалог выбора файла
        filepath = filedialog.askopenfilename(
            title="Open Graph Image",
            initialdir=".",
            filetypes=[
                ("PNG images", "*.png"),
                ("All images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            # Проверяем, что это изображение
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Переключаемся на фрейм изображений
                self.switch_to_image()
                
                # Отображаем изображение
                success = self.image_frame.display_image(filepath)
                if success:
                    filename = os.path.basename(filepath)
                    self.status_var.set(f"📊 Displaying image: {filename}")
                else:
                    self.status_var.set(f"❌ Failed to load image: {filename}")
                    self.switch_to_text()
                    self.text_frame.display_text(f"Error: Could not load image {filename}", None)
            else:
                # Если это не изображение, показываем как текст
                self.switch_to_text()
                self.text_frame.display_file(filepath)
                filename = os.path.basename(filepath)
                self.status_var.set(f"📄 Opened: {filename} (editable)")
                self.save_as_btn.config(state=tk.NORMAL)
    
    def open_file(self):
        """Open arbitrary file for viewing/editing"""
        filepath = filedialog.askopenfilename(
            title="Open File",
            initialdir=".",
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("Output files", "*.out"),
                ("Input files", "*.inp"),
                ("Python files", "*.py"),
                ("CSV files", "*.csv"),
                ("PNG images", "*.png"),
                ("All images", "*.png *.jpg *.jpeg *.gif *.bmp")
            ]
        )
        
        if filepath:
            # Проверяем, изображение ли это
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.switch_to_image()
                success = self.image_frame.display_image(filepath)
                if success:
                    filename = os.path.basename(filepath)
                    self.status_var.set(f"📊 Displaying image: {filename}")
                else:
                    self.status_var.set(f"❌ Failed to load image: {filename}")
                    self.switch_to_text()
                    self.text_frame.display_text(f"Error: Could not load image {filename}", None)
            else:
                self.text_frame.display_file(filepath)
                self.switch_to_text()
                filename = os.path.basename(filepath)
                self.status_var.set(f"📂 Opened: {filename} (editable)")
                # Update save as button state
                self.save_as_btn.config(state=tk.NORMAL)
    
    def read_xout_file(self, filepath: str) -> Optional[PlotData]:
        """Read xout file and return PlotData (удалено, оставлен заглушкой для совместимости)"""
        return None
    
    def switch_to_text(self):
        if self.current_view == 'text':
            return
        self.graph_frame.pack_forget()
        self.image_frame.pack_forget()
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.current_view = 'text'
        self.graph_btn.config(text="📊 Graph")
    
    def switch_to_graph(self):
        if self.current_view == 'graph':
            return
        self.text_frame.pack_forget()
        self.image_frame.pack_forget()
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.current_view = 'graph'
        self.graph_btn.config(text="📊 Graph")
    
    def switch_to_image(self):
        if self.current_view == 'image':
            return
        self.text_frame.pack_forget()
        self.graph_frame.pack_forget()
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.current_view = 'image'
        self.graph_btn.config(text="🖼️ Graph")
    
    def on_save_file(self):
        """Handle Save button or Ctrl+S"""
        if self.current_view == 'text':
            success = self.text_frame.save_file()
            if success:
                self.status_var.set("✅ File saved successfully")
            else:
                if not self.text_frame.get_current_filepath():
                    # No file path - do Save As instead
                    self.on_save_file_as()
    
    def on_save_file_as(self):
        """Handle Save As button"""
        if self.current_view == 'text':
            success = self.text_frame.save_file_as()
            if success:
                self.status_var.set("✅ File saved successfully")
    
    def run(self):
        self.root.mainloop()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    os.makedirs("Inp", exist_ok=True)
    os.makedirs("Out", exist_ok=True)
    
    app = StatisticalApp()
    app.run()