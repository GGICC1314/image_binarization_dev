import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class ImageBinarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像二值化处理器")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # 设置现代化主题
        self.setup_style()
        
        # 初始化变量
        self.original_image = None
        self.backup_original = None  # 备份原始图像用于恢复
        self.grayscale_image = None
        self.binary_image = None
        self.threshold_value = tk.IntVar(value=127)
        self.current_stage = "none"  # none, original, grayscale, binary
        
        # 裁剪相关变量
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None
        self.crop_window = None
        
        # 创建界面
        self.create_widgets()
        
        # 绑定窗口大小变化事件
        self.root.bind('<Configure>', self.on_window_resize)
        
    def setup_style(self):
        """设置现代化UI样式"""
        style = ttk.Style()
        
        # 设置主题
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', 
                       font=('微软雅黑', 16, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Subtitle.TLabel', 
                       font=('微软雅黑', 12),
                       foreground='#34495e')
        
        style.configure('Info.TLabel', 
                       font=('微软雅黑', 10),
                       foreground='#7f8c8d')
        
        style.configure('Modern.TButton',
                       font=('微软雅黑', 10),
                       padding=(20, 10))
        
        style.configure('Modern.TCheckbutton',
                       font=('微软雅黑', 10))
        
        # 修复Scale样式配置
        style.configure('TScale',
                       troughcolor='#ecf0f1',
                       background='#3498db',
                       lightcolor='#3498db',
                       darkcolor='#2980b9')
        
    def create_widgets(self):
        """创建主要UI组件"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重使界面可以缩放
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # 图像显示区域权重最大
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像二值化处理器", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # 控制面板 - 使用滚动框架
        self.create_scrollable_control_panel(main_frame)
        
        # 图像显示区域（支持多图像并排显示）
        self.create_image_display_area(main_frame)
    
    def create_scrollable_control_panel(self, parent):
        """创建可滚动的控制面板"""
        # 控制面板外框
        control_outer_frame = ttk.Frame(parent)
        control_outer_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_outer_frame.columnconfigure(0, weight=1)
        
        # 创建画布和滚动条
        canvas = tk.Canvas(control_outer_frame, height=250, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # 在可滚动框架中创建控制面板
        self.create_control_panel(scrollable_frame)
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.Frame(parent, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        
        # 操作按钮区域
        button_frame = ttk.LabelFrame(control_frame, text="图像处理步骤", padding="10")
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        # 第一行按钮
        self.import_btn = ttk.Button(button_frame, text="1. 导入图片", 
                                    command=self.import_image, style='Modern.TButton')
        self.import_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=(0, 5))
        
        self.grayscale_btn = ttk.Button(button_frame, text="2. 转换灰度图", 
                                       command=self.convert_to_grayscale, 
                                       style='Modern.TButton', state='disabled')
        self.grayscale_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(0, 5))
        
        # 第二行按钮
        self.binary_btn = ttk.Button(button_frame, text="3. 启用二值化", 
                                    command=self.enable_binarization, 
                                    style='Modern.TButton', state='disabled')
        self.binary_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.save_btn = ttk.Button(button_frame, text="4. 保存结果", 
                                  command=self.save_result, 
                                  style='Modern.TButton', state='disabled')
        self.save_btn.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # 图像编辑功能区域
        edit_frame = ttk.LabelFrame(control_frame, text="图像编辑功能", padding="10")
        edit_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        edit_frame.columnconfigure(0, weight=1)
        edit_frame.columnconfigure(1, weight=1)
        
        # 裁剪图像按钮
        self.crop_btn = ttk.Button(edit_frame, text="裁剪图像", 
                                  command=self.start_crop_mode, 
                                  style='Modern.TButton', state='disabled')
        self.crop_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # 恢复原图按钮
        self.restore_btn = ttk.Button(edit_frame, text="恢复原图", 
                                     command=self.restore_original, 
                                     style='Modern.TButton', state='disabled')
        self.restore_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # 阈值调整区域
        threshold_frame = ttk.LabelFrame(control_frame, text="二值化阈值调整", padding="10")
        threshold_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        threshold_frame.columnconfigure(1, weight=1)
        
        # 像素统计区域
        stats_frame = ttk.LabelFrame(control_frame, text="像素统计", padding="10")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)
        
        # 统计按钮
        self.stats_btn = ttk.Button(stats_frame, text="📊 统计像素数量", 
                                   command=self.calculate_pixel_statistics, 
                                   style='Modern.TButton', state='disabled')
        self.stats_btn.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 黑色像素统计
        ttk.Label(stats_frame, text="黑色像素点数量:", style='Subtitle.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.black_pixels_label = ttk.Label(stats_frame, text="--", 
                                           style='Info.TLabel', font=('微软雅黑', 11, 'bold'),
                                           foreground='#2c3e50')
        self.black_pixels_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 白色像素统计
        ttk.Label(stats_frame, text="白色像素点数量:", style='Subtitle.TLabel').grid(row=2, column=0, sticky=tk.W)
        self.white_pixels_label = ttk.Label(stats_frame, text="--", 
                                           style='Info.TLabel', font=('微软雅黑', 11, 'bold'),
                                           foreground='#2c3e50')
        self.white_pixels_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # 总像素数
        ttk.Label(stats_frame, text="总像素数:", style='Subtitle.TLabel').grid(row=3, column=0, sticky=tk.W)
        self.total_pixels_label = ttk.Label(stats_frame, text="--", style='Info.TLabel')
        self.total_pixels_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        # 黑白比例
        ttk.Label(stats_frame, text="黑白比例:", style='Subtitle.TLabel').grid(row=4, column=0, sticky=tk.W)
        self.ratio_label = ttk.Label(stats_frame, text="--", style='Info.TLabel')
        self.ratio_label.grid(row=4, column=1, sticky=tk.W, padx=(10, 0))
        
        # 阈值说明
        self.threshold_info = ttk.Label(threshold_frame, text="请先完成前面的步骤", 
                                       style='Info.TLabel', foreground='#95a5a6')
        self.threshold_info.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # 阈值标签和数值显示
        ttk.Label(threshold_frame, text="阈值:", style='Subtitle.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.threshold_label = ttk.Label(threshold_frame, text="127", style='Subtitle.TLabel')
        self.threshold_label.grid(row=1, column=2, sticky=tk.E)
        
        # 阈值滑块
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0, to=255, 
                                        variable=self.threshold_value,
                                        orient=tk.HORIZONTAL,
                                        command=self.on_threshold_change,
                                        state='disabled')
        self.threshold_scale.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_image_display_area(self, parent):
        """创建图像显示区域"""
        display_frame = ttk.LabelFrame(parent, text="图像处理过程", padding="10")
        display_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 原图显示
        self.create_image_panel(display_frame, 0, "原图", "original")
        
        # 灰度图显示
        self.create_image_panel(display_frame, 1, "灰度图", "grayscale")
        
        # 二值化图显示
        self.create_image_panel(display_frame, 2, "二值化结果", "binary")
    
    def create_image_panel(self, parent, column, title, panel_type):
        """创建单个图像显示面板"""
        # 面板容器
        panel_frame = ttk.LabelFrame(parent, text=title, padding="10")
        panel_frame.grid(row=0, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        panel_frame.columnconfigure(0, weight=1)
        panel_frame.rowconfigure(0, weight=1)
        
        # 创建画布
        canvas = tk.Canvas(panel_frame, bg='#f8f9fa', relief=tk.SUNKEN, bd=1, height=400)
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图像信息标签
        info_label = ttk.Label(panel_frame, text="暂无图像", 
                              style='Info.TLabel', anchor='center')
        info_label.grid(row=1, column=0, pady=(5, 0))
        
        # 存储画布和信息标签的引用
        if panel_type == "original":
            self.original_canvas = canvas
            self.original_info = info_label
        elif panel_type == "grayscale":
            self.grayscale_canvas = canvas
            self.grayscale_info = info_label
        elif panel_type == "binary":
            self.binary_canvas = canvas
            self.binary_info = info_label
        
    def import_image(self):
        """导入图片"""
        file_types = [
            ('图片文件', '*.png *.jpg *.jpeg *.bmp *.tiff *.gif'),
            ('PNG文件', '*.png'),
            ('JPEG文件', '*.jpg *.jpeg'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # 使用OpenCV读取图像
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("错误", "无法读取图片文件！")
                    return
                
                # 转换颜色空间（OpenCV使用BGR，PIL使用RGB）
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # 备份原始图像
                self.backup_original = self.original_image.copy()
                
                # 重置状态
                self.grayscale_image = None
                self.binary_image = None
                self.current_stage = "original"
                
                # 清除其他画布
                self.clear_canvas(self.grayscale_canvas)
                self.clear_canvas(self.binary_canvas)
                self.grayscale_info.config(text="暂无图像")
                self.binary_info.config(text="暂无图像")
                
                # 显示原图
                self.display_image_on_canvas(self.original_image, self.original_canvas, self.original_info)
                
                # 更新按钮状态
                self.update_button_states()
                
            except Exception as e:
                messagebox.showerror("错误", f"读取图片时发生错误: {str(e)}")
    
    def convert_to_grayscale(self):
        """转换为灰度图"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先导入图片！")
            return
        
        try:
            # 检查是否已经是灰度图
            if len(self.original_image.shape) == 2:
                self.grayscale_image = self.original_image.copy()
            else:
                # 转换为灰度图
                self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            
            # 更新状态
            self.current_stage = "grayscale"
            
            # 显示灰度图
            self.display_image_on_canvas(self.grayscale_image, self.grayscale_canvas, self.grayscale_info)
            
            # 更新按钮状态
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror("错误", f"转换灰度图时发生错误: {str(e)}")
    
    def enable_binarization(self):
        """启用二值化功能"""
        if self.grayscale_image is None:
            messagebox.showwarning("警告", "请先转换为灰度图！")
            return
        
        # 更新状态
        self.current_stage = "binary"
        
        # 启用阈值滑块
        self.threshold_scale.config(state='normal')
        self.threshold_info.config(text="拖动滑块调整阈值，实时查看二值化效果", foreground='#27ae60')
        
        # 执行初始二值化
        self.process_binary_image()
        
        # 更新按钮状态
        self.update_button_states()
    
    def start_crop_mode(self):
        """开始裁剪模式"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先导入图片！")
            return
        
        # 创建裁剪窗口
        self.create_crop_window()
    
    def create_crop_window(self):
        """创建裁剪窗口"""
        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("图像裁剪 - 拖拽选择裁剪区域")
        self.crop_window.geometry("800x700")
        self.crop_window.transient(self.root)
        self.crop_window.grab_set()
        
        # 创建框架
        main_frame = ttk.Frame(self.crop_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明标签
        info_label = ttk.Label(main_frame, text="拖拽鼠标选择要裁剪的矩形区域，然后点击确认裁剪", 
                              style='Subtitle.TLabel')
        info_label.pack(pady=(0, 10))
        
        # 创建画布
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.crop_canvas = tk.Canvas(canvas_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.crop_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.crop_canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.crop_canvas.xview)
        h_scroll.pack(fill=tk.X)
        
        self.crop_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 确认裁剪按钮
        confirm_btn = ttk.Button(button_frame, text="确认裁剪", 
                               command=self.confirm_crop, style='Modern.TButton')
        confirm_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 取消按钮
        cancel_btn = ttk.Button(button_frame, text="取消", 
                              command=self.cancel_crop, style='Modern.TButton')
        cancel_btn.pack(side=tk.LEFT)
        
        # 显示原图
        self.display_crop_image()
        
        # 绑定鼠标事件
        self.crop_canvas.bind('<Button-1>', self.start_crop_selection)
        self.crop_canvas.bind('<B1-Motion>', self.update_crop_selection)
        self.crop_canvas.bind('<ButtonRelease-1>', self.end_crop_selection)
        
        # 重置裁剪变量
        self.crop_start = None
        self.crop_rect = None
    
    def display_crop_image(self):
        """在裁剪窗口显示原图"""
        if self.original_image is None:
            return
        
        # 获取画布大小
        self.crop_window.update()
        canvas_width = self.crop_canvas.winfo_width()
        canvas_height = self.crop_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.crop_window.after(100, self.display_crop_image)
            return
        
        # 计算缩放比例（稍微放大以便精确选择）
        img_height, img_width = self.original_image.shape[:2]
        scale_x = (canvas_width - 40) / img_width
        scale_y = (canvas_height - 40) / img_height
        self.crop_scale = min(scale_x, scale_y, 2.0)  # 最大放大2倍
        
        new_width = int(img_width * self.crop_scale)
        new_height = int(img_height * self.crop_scale)
        
        # 调整图像大小
        resized_image = cv2.resize(self.original_image, (new_width, new_height), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(resized_image)
        self.crop_photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图像
        self.crop_canvas.delete("all")
        
        # 居中显示图像
        self.crop_offset_x = (canvas_width - new_width) // 2
        self.crop_offset_y = (canvas_height - new_height) // 2
        
        self.crop_canvas.create_image(self.crop_offset_x, self.crop_offset_y, 
                                    anchor=tk.NW, image=self.crop_photo)
        
        # 更新滚动区域
        self.crop_canvas.configure(scrollregion=self.crop_canvas.bbox("all"))
    
    def start_crop_selection(self, event):
        """开始裁剪选择"""
        # 转换画布坐标
        canvas_x = self.crop_canvas.canvasx(event.x)
        canvas_y = self.crop_canvas.canvasy(event.y)
        
        # 检查是否在图像范围内
        if (canvas_x >= self.crop_offset_x and 
            canvas_y >= self.crop_offset_y and
            canvas_x <= self.crop_offset_x + self.crop_photo.width() and
            canvas_y <= self.crop_offset_y + self.crop_photo.height()):
            
            self.crop_start = (canvas_x, canvas_y)
            # 删除之前的矩形
            if self.crop_rect:
                self.crop_canvas.delete(self.crop_rect)
    
    def update_crop_selection(self, event):
        """更新裁剪选择"""
        if self.crop_start is None:
            return
        
        # 转换画布坐标
        canvas_x = self.crop_canvas.canvasx(event.x)
        canvas_y = self.crop_canvas.canvasy(event.y)
        
        # 限制在图像范围内
        canvas_x = max(self.crop_offset_x, min(canvas_x, self.crop_offset_x + self.crop_photo.width()))
        canvas_y = max(self.crop_offset_y, min(canvas_y, self.crop_offset_y + self.crop_photo.height()))
        
        # 删除之前的矩形
        if self.crop_rect:
            self.crop_canvas.delete(self.crop_rect)
        
        # 绘制新的矩形
        self.crop_rect = self.crop_canvas.create_rectangle(
            self.crop_start[0], self.crop_start[1], canvas_x, canvas_y,
            outline='red', width=2, dash=(5, 5)
        )
    
    def end_crop_selection(self, event):
        """结束裁剪选择"""
        pass  # 保持矩形显示
    
    def confirm_crop(self):
        """确认裁剪"""
        if self.crop_start is None or self.crop_rect is None:
            messagebox.showwarning("警告", "请先选择裁剪区域！")
            return
        
        try:
            # 获取矩形坐标
            coords = self.crop_canvas.coords(self.crop_rect)
            x1, y1, x2, y2 = coords
            
            # 确保坐标顺序正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 转换为原图坐标
            orig_x1 = int((x1 - self.crop_offset_x) / self.crop_scale)
            orig_y1 = int((y1 - self.crop_offset_y) / self.crop_scale)
            orig_x2 = int((x2 - self.crop_offset_x) / self.crop_scale)
            orig_y2 = int((y2 - self.crop_offset_y) / self.crop_scale)
            
            # 确保坐标在图像范围内
            img_height, img_width = self.original_image.shape[:2]
            orig_x1 = max(0, min(orig_x1, img_width-1))
            orig_y1 = max(0, min(orig_y1, img_height-1))
            orig_x2 = max(0, min(orig_x2, img_width))
            orig_y2 = max(0, min(orig_y2, img_height))
            
            # 检查裁剪区域大小
            if orig_x2 - orig_x1 < 10 or orig_y2 - orig_y1 < 10:
                messagebox.showwarning("警告", "裁剪区域太小！")
                return
            
            # 执行裁剪
            self.original_image = self.original_image[orig_y1:orig_y2, orig_x1:orig_x2]
            
            # 重置后续处理结果
            self.grayscale_image = None
            self.binary_image = None
            self.current_stage = "original"
            
            # 关闭裁剪窗口
            self.cancel_crop()
            
            # 更新主界面显示
            self.clear_canvas(self.grayscale_canvas)
            self.clear_canvas(self.binary_canvas)
            self.grayscale_info.config(text="暂无图像")
            self.binary_info.config(text="暂无图像")
            
            # 显示裁剪后的图像
            self.display_image_on_canvas(self.original_image, self.original_canvas, self.original_info)
            
            # 更新按钮状态
            self.update_button_states()
            
            # 重置阈值滑块
            self.threshold_scale.config(state='disabled')
            self.threshold_info.config(text="请先完成前面的步骤", foreground='#95a5a6')
            self.update_pixel_stats()
            
            messagebox.showinfo("成功", "图像裁剪完成！")
            
        except Exception as e:
            messagebox.showerror("错误", f"裁剪图像时发生错误: {str(e)}")
    
    def cancel_crop(self):
        """取消裁剪"""
        if self.crop_window:
            self.crop_window.destroy()
            self.crop_window = None
        self.crop_start = None
        self.crop_rect = None
    
    def restore_original(self):
        """恢复原图"""
        if self.backup_original is None:
            messagebox.showwarning("警告", "没有可恢复的原图！")
            return
        
        try:
            # 恢复原图
            self.original_image = self.backup_original.copy()
            
            # 重置状态
            self.grayscale_image = None
            self.binary_image = None
            self.current_stage = "original"
            
            # 清除其他画布
            self.clear_canvas(self.grayscale_canvas)
            self.clear_canvas(self.binary_canvas)
            self.grayscale_info.config(text="暂无图像")
            self.binary_info.config(text="暂无图像")
            
            # 显示原图
            self.display_image_on_canvas(self.original_image, self.original_canvas, self.original_info)
            
            # 更新按钮状态
            self.update_button_states()
            
            # 重置阈值滑块和统计
            self.threshold_scale.config(state='disabled')
            self.threshold_info.config(text="请先完成前面的步骤", foreground='#95a5a6')
            self.update_pixel_stats()
            
            messagebox.showinfo("成功", "原图已恢复！")
            
        except Exception as e:
            messagebox.showerror("错误", f"恢复原图时发生错误: {str(e)}")
    
    def on_threshold_change(self, value):
        """阈值改变时的处理（实时更新）"""
        threshold = int(float(value))
        self.threshold_label.config(text=str(threshold))
        
        if self.current_stage == "binary" and self.grayscale_image is not None:
            self.process_binary_image()
    
    def process_binary_image(self):
        """处理二值化图像"""
        if self.grayscale_image is None:
            return
        
        try:
            # 应用二值化
            threshold = self.threshold_value.get()
            _, binary_image = cv2.threshold(self.grayscale_image, threshold, 255, cv2.THRESH_BINARY)
            
            self.binary_image = binary_image
            
            # 显示二值化结果
            self.display_image_on_canvas(self.binary_image, self.binary_canvas, self.binary_info)
            
            # 更新像素统计
            self.update_pixel_stats()
            
        except Exception as e:
            messagebox.showerror("错误", f"处理二值化图像时发生错误: {str(e)}")
    
    def display_image_on_canvas(self, image, canvas, info_label):
        """在指定画布上显示图像"""
        if image is None:
            return
        
        try:
            # 获取画布大小
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.root.after(100, lambda: self.display_image_on_canvas(image, canvas, info_label))
                return
            
            # 计算缩放比例以适应画布
            if len(image.shape) == 3:
                img_height, img_width = image.shape[:2]
            else:
                img_height, img_width = image.shape
            
            scale_x = (canvas_width - 20) / img_width
            scale_y = (canvas_height - 20) / img_height
            scale = min(scale_x, scale_y, 1.0)  # 不放大，只缩小
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图像大小
            if new_width > 0 and new_height > 0:
                resized_image = cv2.resize(image, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
                
                # 转换为PIL图像
                if len(resized_image.shape) == 2:
                    # 灰度图或二值图
                    pil_image = Image.fromarray(resized_image)
                else:
                    # 彩色图
                    pil_image = Image.fromarray(resized_image)
                
                photo = ImageTk.PhotoImage(pil_image)
                
                # 清除画布并显示新图像
                canvas.delete("all")
                
                # 居中显示图像
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                
                canvas.create_image(x, y, anchor=tk.NW, image=photo)
                
                # 保存图像引用以防被垃圾回收
                canvas.image = photo
                
                # 更新信息标签
                info_text = f"{img_width} × {img_height}"
                if len(image.shape) == 2:
                    info_text += " (灰度)"
                else:
                    info_text += f" ({image.shape[2]}通道)"
                info_label.config(text=info_text)
            
        except Exception as e:
            print(f"显示图像时发生错误: {str(e)}")
    
    def clear_canvas(self, canvas):
        """清除画布内容"""
        canvas.delete("all")
        if hasattr(canvas, 'image'):
            delattr(canvas, 'image')
    
    def update_button_states(self):
        """更新按钮状态"""
        if self.current_stage == "none":
            self.grayscale_btn.config(state='disabled')
            self.binary_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            self.crop_btn.config(state='disabled')
            self.restore_btn.config(state='disabled')
            self.stats_btn.config(state='disabled')
        elif self.current_stage == "original":
            self.grayscale_btn.config(state='normal')
            self.binary_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            self.crop_btn.config(state='normal')
            self.restore_btn.config(state='normal')
            self.stats_btn.config(state='disabled')
        elif self.current_stage == "grayscale":
            self.grayscale_btn.config(state='disabled')
            self.binary_btn.config(state='normal')
            self.save_btn.config(state='disabled')
            self.crop_btn.config(state='normal')
            self.restore_btn.config(state='normal')
            self.stats_btn.config(state='disabled')
        elif self.current_stage == "binary":
            self.grayscale_btn.config(state='disabled')
            self.binary_btn.config(state='disabled')
            self.save_btn.config(state='normal')
            self.crop_btn.config(state='normal')
            self.restore_btn.config(state='normal')
            self.stats_btn.config(state='normal')
    
    def update_pixel_stats(self):
        """自动更新像素统计信息（内部调用）"""
        if self.binary_image is None:
            # 清空统计信息
            self.black_pixels_label.config(text="--")
            self.white_pixels_label.config(text="--")
            self.total_pixels_label.config(text="--")
            self.ratio_label.config(text="--")
            return
        
        # 自动统计（不显示详细信息）
        try:
            total_pixels = self.binary_image.size
            self.total_pixels_label.config(text=f"{total_pixels:,}")
        except Exception as e:
            print(f"更新像素统计时发生错误: {str(e)}")
    
    def calculate_pixel_statistics(self):
        """计算并显示详细的像素统计信息（按钮触发）"""
        if self.binary_image is None:
            messagebox.showwarning("警告", "请先完成二值化处理！")
            return
        
        try:
            # 统计黑白像素
            total_pixels = self.binary_image.size
            white_pixels = np.sum(self.binary_image == 255)
            black_pixels = np.sum(self.binary_image == 0)
            
            # 计算比例
            if total_pixels > 0:
                black_ratio = (black_pixels / total_pixels) * 100
                white_ratio = (white_pixels / total_pixels) * 100
                
                # 更新显示
                self.black_pixels_label.config(text=f"{black_pixels:,} ({black_ratio:.1f}%)")
                self.white_pixels_label.config(text=f"{white_pixels:,} ({white_ratio:.1f}%)")
                self.total_pixels_label.config(text=f"{total_pixels:,}")
                
                if black_pixels > 0 and white_pixels > 0:
                    ratio = white_pixels / black_pixels
                    self.ratio_label.config(text=f"白:黑 = {ratio:.2f}:1")
                else:
                    self.ratio_label.config(text="N/A")
                
                # 显示详细统计信息对话框
                stats_info = f"""二值化图像像素统计结果：

📊 总像素数：{total_pixels:,} 个像素点

⚫ 黑色像素点数量：{black_pixels:,} 个
   占比：{black_ratio:.2f}%

⚪ 白色像素点数量：{white_pixels:,} 个  
   占比：{white_ratio:.2f}%

📈 黑白比例：白色 : 黑色 = {ratio:.2f} : 1"""
                
                messagebox.showinfo("像素统计结果", stats_info)
                
            else:
                self.black_pixels_label.config(text="0")
                self.white_pixels_label.config(text="0")
                self.total_pixels_label.config(text="0")
                self.ratio_label.config(text="N/A")
                
        except Exception as e:
            messagebox.showerror("错误", f"统计像素时发生错误: {str(e)}")
            self.black_pixels_label.config(text="错误")
            self.white_pixels_label.config(text="错误")
            self.total_pixels_label.config(text="错误")
            self.ratio_label.config(text="错误")
    
    def save_result(self):
        """保存处理结果"""
        if self.binary_image is None:
            messagebox.showwarning("警告", "没有可保存的二值化图像！")
            return
        
        file_types = [
            ('PNG文件', '*.png'),
            ('JPEG文件', '*.jpg'),
            ('BMP文件', '*.bmp'),
            ('TIFF文件', '*.tiff')
        ]
        
        file_path = filedialog.asksaveasfilename(
            title="保存二值化结果",
            defaultextension=".png",
            filetypes=file_types
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.binary_image)
                messagebox.showinfo("成功", "二值化图像保存成功！")
            except Exception as e:
                messagebox.showerror("错误", f"保存图像时发生错误: {str(e)}")
    
    def on_window_resize(self, event):
        """窗口大小改变时的处理"""
        if event.widget == self.root:
            # 延迟更新显示以避免频繁刷新
            self.root.after(200, self.refresh_all_images)
    
    def refresh_all_images(self):
        """刷新所有图像显示"""
        if self.original_image is not None:
            self.display_image_on_canvas(self.original_image, self.original_canvas, self.original_info)
        
        if self.grayscale_image is not None:
            self.display_image_on_canvas(self.grayscale_image, self.grayscale_canvas, self.grayscale_info)
        
        if self.binary_image is not None:
            self.display_image_on_canvas(self.binary_image, self.binary_canvas, self.binary_info)

def main():
    """主函数"""
    root = tk.Tk()
    app = ImageBinarizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
