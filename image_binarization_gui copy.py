#version 0.1
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
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 设置现代化主题
        self.setup_style()
        
        # 初始化变量
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        self.threshold_value = tk.IntVar(value=127)
        self.is_grayscale = tk.BooleanVar(value=False)
        
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
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重使界面可以缩放
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像二值化处理器", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧图像显示区域
        self.create_image_panel(main_frame)
        
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.columnconfigure(0, weight=1)
        
        # 导入图片按钮
        import_btn = ttk.Button(file_frame, text="导入图片", 
                               command=self.import_image, style='Modern.TButton')
        import_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 保存结果按钮
        save_btn = ttk.Button(file_frame, text="保存结果", 
                             command=self.save_result, style='Modern.TButton')
        save_btn.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 图像处理选项
        process_frame = ttk.LabelFrame(control_frame, text="图像处理", padding="10")
        process_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        process_frame.columnconfigure(0, weight=1)
        
        # 灰度转换选项
        self.grayscale_cb = ttk.Checkbutton(process_frame, text="转换为灰度图", 
                                           variable=self.is_grayscale,
                                           command=self.on_grayscale_change,
                                           style='Modern.TCheckbutton')
        self.grayscale_cb.grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        # 阈值调整区域
        threshold_frame = ttk.LabelFrame(control_frame, text="二值化阈值", padding="10")
        threshold_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        threshold_frame.columnconfigure(1, weight=1)
        
        # 阈值标签和数值显示
        ttk.Label(threshold_frame, text="阈值:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.threshold_label = ttk.Label(threshold_frame, text="127", style='Subtitle.TLabel')
        self.threshold_label.grid(row=0, column=2, sticky=tk.E)
        
        # 阈值滑块
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0, to=255, 
                                        variable=self.threshold_value,
                                        orient=tk.HORIZONTAL,
                                        command=self.on_threshold_change)
        self.threshold_scale.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 图像信息显示
        info_frame = ttk.LabelFrame(control_frame, text="图像信息", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(1, weight=1)
        
        self.info_text = tk.Text(info_frame, height=6, width=30, wrap=tk.WORD,
                                font=('微软雅黑', 9), state=tk.DISABLED,
                                bg='#f8f9fa', relief=tk.FLAT)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.config(yscrollcommand=info_scrollbar.set)
        
    def create_image_panel(self, parent):
        """创建右侧图像显示面板"""
        image_frame = ttk.LabelFrame(parent, text="图像预览", padding="15")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # 创建画布用于显示图像
        self.canvas = tk.Canvas(image_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加滚动条
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
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
                
                # 检查是否为灰度图
                if len(self.original_image.shape) == 3:
                    self.is_grayscale.set(False)
                else:
                    self.is_grayscale.set(True)
                
                self.current_image = self.original_image.copy()
                self.update_image_info()
                self.process_image()
                
            except Exception as e:
                messagebox.showerror("错误", f"读取图片时发生错误: {str(e)}")
    
    def on_grayscale_change(self):
        """灰度转换选项改变时的处理"""
        if self.original_image is not None:
            if self.is_grayscale.get():
                # 转换为灰度图
                if len(self.original_image.shape) == 3:
                    self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                else:
                    self.current_image = self.original_image.copy()
            else:
                # 使用原图
                self.current_image = self.original_image.copy()
            
            self.process_image()
    
    def on_threshold_change(self, value):
        """阈值改变时的处理（实时更新）"""
        threshold = int(float(value))
        self.threshold_label.config(text=str(threshold))
        
        if self.current_image is not None:
            self.process_image()
    
    def process_image(self):
        """处理图像（二值化）"""
        if self.current_image is None:
            return
        
        try:
            # 确保图像是灰度图
            if len(self.current_image.shape) == 3:
                gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.current_image.copy()
            
            # 应用二值化
            threshold = self.threshold_value.get()
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            
            self.processed_image = binary_image
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("错误", f"处理图像时发生错误: {str(e)}")
    
    def display_image(self):
        """在画布上显示图像"""
        if self.processed_image is None:
            return
        
        try:
            # 获取画布大小
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.root.after(100, self.display_image)
                return
            
            # 计算缩放比例以适应画布
            img_height, img_width = self.processed_image.shape
            scale_x = (canvas_width - 20) / img_width
            scale_y = (canvas_height - 20) / img_height
            scale = min(scale_x, scale_y, 1.0)  # 不放大，只缩小
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图像大小
            resized_image = cv2.resize(self.processed_image, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(resized_image)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # 清除画布并显示新图像
            self.canvas.delete("all")
            
            # 居中显示图像
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
            # 更新滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"显示图像时发生错误: {str(e)}")
    
    def update_image_info(self):
        """更新图像信息显示"""
        if self.original_image is None:
            return
        
        try:
            info = []
            if len(self.original_image.shape) == 3:
                height, width, channels = self.original_image.shape
                info.append(f"尺寸: {width} × {height}")
                info.append(f"通道数: {channels}")
                info.append(f"类型: 彩色图像")
            else:
                height, width = self.original_image.shape
                info.append(f"尺寸: {width} × {height}")
                info.append(f"类型: 灰度图像")
            
            info.append(f"数据类型: {self.original_image.dtype}")
            info.append(f"像素范围: {self.original_image.min()} - {self.original_image.max()}")
            
            # 更新信息显示
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, "\n".join(info))
            self.info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"更新图像信息时发生错误: {str(e)}")
    
    def save_result(self):
        """保存处理结果"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有可保存的图像！")
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
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("成功", "图像保存成功！")
            except Exception as e:
                messagebox.showerror("错误", f"保存图像时发生错误: {str(e)}")
    
    def on_window_resize(self, event):
        """窗口大小改变时的处理"""
        if event.widget == self.root:
            # 延迟更新显示以避免频繁刷新
            self.root.after(100, self.display_image)

def main():
    """主函数"""
    root = tk.Tk()
    app = ImageBinarizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
