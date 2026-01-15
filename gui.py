# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2023/4/1 6:07 PM
@FileName: gui.py
@desc: Subtitle Remover GUI
"""
import os
import configparser
import PySimpleGUI as sg
import cv2
import sys
from threading import Thread
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.main
from backend.tools.common_tools import is_image_file


class SubtitleRemoverGUI:

    def __init__(self):
        self.font = 'Arial 10'
        self.theme = 'LightBrown12'
        sg.theme(self.theme)
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vsr.ico')
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        print(self.screen_width, self.screen_height)
        # Set video preview area size
        self.video_preview_width = 960
        self.video_preview_height = self.video_preview_width * 9 // 16
        # Default component size
        self.horizontal_slider_size = (120, 20)
        self.output_size = (100, 10)
        self.progressbar_size = (60, 20)
        # Resolution lower than 1080
        if self.screen_width // 2 < 960:
            self.video_preview_width = 640
            self.video_preview_height = self.video_preview_width * 9 // 16
            self.horizontal_slider_size = (60, 20)
            self.output_size = (58, 10)
            self.progressbar_size = (28, 20)
        # Subtitle extractor layout
        self.layout = None
        # Subtitle extractor window
        self.window = None
        # Video path
        self.video_path = None
        # Video capture
        self.video_cap = None
        # Video FPS
        self.fps = None
        # Video frame count
        self.frame_count = None
        # Video width
        self.frame_width = None
        # Video height
        self.frame_height = None
        # Set subtitle area height and width
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        # Subtitle remover
        self.sr = None

    def run(self):
        # Create layout
        self._create_layout()
        # Create window
        self.window = sg.Window(title=f'Video Subtitle Remover v{backend.main.config.VERSION}' , layout=self.layout,
                                icon=self.icon)
        while True:
            # Loop to read events
            event, values = self.window.read(timeout=10)
            # Handle [Open] event
            self._file_event_handler(event, values)
            # Handle [Slide] event
            self._slide_event_handler(event, values)
            # Handle [Run] event
            self._run_event_handler(event, values)
            # If closing software, exit
            if event == sg.WIN_CLOSED:
                break
            # Update progress bar
            if self.sr is not None:
                self.window['-PROG-'].update(self.sr.progress_total)
                if self.sr.preview_frame is not None:
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', self._img_resize(self.sr.preview_frame))[1].tobytes())
                if self.sr.isFinished:
                    # 1) Enable subtitle slider area modification
                    self.window['-Y-SLIDER-'].update(disabled=False)
                    self.window['-X-SLIDER-'].update(disabled=False)
                    self.window['-Y-SLIDER-H-'].update(disabled=False)
                    self.window['-X-SLIDER-W-'].update(disabled=False)
                    # 2) Enable [Run], [Open], and [Identify Language] buttons
                    self.window['-RUN-'].update(disabled=False)
                    self.window['-FILE-'].update(disabled=False)
                    self.window['-FILE_BTN-'].update(disabled=False)
                    self.sr = None
                if len(self.video_paths) >= 1:
                    # 1) Disable subtitle slider area modification
                    self.window['-Y-SLIDER-'].update(disabled=True)
                    self.window['-X-SLIDER-'].update(disabled=True)
                    self.window['-Y-SLIDER-H-'].update(disabled=True)
                    self.window['-X-SLIDER-W-'].update(disabled=True)
                    # 2) Disable [Run], [Open], and [Identify Language] buttons
                    self.window['-RUN-'].update(disabled=True)
                    self.window['-FILE-'].update(disabled=True)
                    self.window['-FILE_BTN-'].update(disabled=True)

    def _create_layout(self):
        """
        Create Subtitle Remover Layout
        """
        garbage = os.path.join(os.path.dirname(__file__), 'output')
        if os.path.exists(garbage):
            import shutil
            shutil.rmtree(garbage, True)
        self.layout = [
            # Display video preview
            [sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='black',
                      key='-DISPLAY-')],
            # Open button + fast forward/rewind slider
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FilesBrowse(button_text='Open', file_types=((
                            'All Files', '*.*'), ('mp4', '*.mp4'),
                            ('flv', '*.flv'),
                            ('wmv', '*.wmv'),
                            ('avi', '*.avi')),
                            key='-FILE_BTN-', size=(10, 1), font=self.font),
             sg.Slider(size=self.horizontal_slider_size, range=(1, 1), key='-SLIDER-', orientation='h',
                       enable_events=True, font=self.font,
                       disable_number_display=True),
             ],
            # Output area
            [sg.Output(size=self.output_size, font=self.font),
             sg.Frame(title='Vertical', font=self.font, key='-FRAME1-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((15, 5), (0, 0))),
             sg.Frame(title='Horizontal', font=self.font, key='-FRAME2-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-W-'),
             ]], pad=((15, 5), (0, 0)))
             ],

            # Run button + progress bar
            [sg.Button(button_text='Run', key='-RUN-',
                       font=self.font, size=(20, 1)),
             sg.ProgressBar(100, orientation='h', size=self.progressbar_size, key='-PROG-', auto_size_text=True)
             ],
        ]

    def _file_event_handler(self, event, values):
        """
        When the Open button is clicked:
        1) Open the video file and display the video frame on the canvas
        2) Get video information and initialize the progress bar slider range
        """
        if event == '-FILE-':
            self.video_paths = values['-FILE-'].split(';')
            self.video_path = self.video_paths[0]
            if self.video_path != '':
                self.video_cap = cv2.VideoCapture(self.video_path)
            if self.video_cap is None:
                return
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    for video in self.video_paths:
                        print(f"Open Video Success：{video}")
                    # Get video frame count
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # Get video height
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # Get video width
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # Get video FPS
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    # Resize video frame for player display
                    resized_frame = self._img_resize(frame)
                    # resized_frame = cv2.resize(src=frame, dsize=(self.video_preview_width, self.video_preview_height))
                    # Display video frame
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    # Update video progress slider range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # Preset subtitle area position
                    y_p, h_p, x_p, w_p = self.parse_subtitle_config()
                    y = self.frame_height * y_p
                    h = self.frame_height * h_p
                    x = self.frame_width * x_p
                    w = self.frame_width * w_p
                    # Update video subtitle position slider range
                    # Update Y-SLIDER range
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    # Update Y-SLIDER default value
                    self.window['-Y-SLIDER-'].update(y)
                    # Update X-SLIDER range
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    # Update X-SLIDER default value
                    self.window['-X-SLIDER-'].update(x)
                    # Update Y-SLIDER-H range
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - y))
                    # Update Y-SLIDER-H default value
                    self.window['-Y-SLIDER-H-'].update(h)
                    # Update X-SLIDER-W range
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - x))
                    # Update X-SLIDER-W default value
                    self.window['-X-SLIDER-W-'].update(w)
                    self._update_preview(frame, (y, h, x, w))

    def __disable_button(self):
        # 1) Disable subtitle slider area modification
        self.window['-Y-SLIDER-'].update(disabled=True)
        self.window['-X-SLIDER-'].update(disabled=True)
        self.window['-Y-SLIDER-H-'].update(disabled=True)
        self.window['-X-SLIDER-W-'].update(disabled=True)
        # 2) Disable clicking [Run], [Open], and [Identify Language] buttons again
        self.window['-RUN-'].update(disabled=True)
        self.window['-FILE-'].update(disabled=True)
        self.window['-FILE_BTN-'].update(disabled=True)

    def _run_event_handler(self, event, values):
        """
        When the Run button is clicked:
        1) Disable subtitle slider area modification
        2) Disable clicking [Run] and [Open] buttons again
        3) Set subtitle area position
        """
        if event == '-RUN-':
            if self.video_cap is None:
                print('Please Open Video First')
            else:
                # Disable buttons
                self.__disable_button()
                # 3) Set subtitle area position
                self.xmin = int(values['-X-SLIDER-'])
                self.xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
                self.ymin = int(values['-Y-SLIDER-'])
                self.ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
                if self.ymax > self.frame_height:
                    self.ymax = self.frame_height
                if self.xmax > self.frame_width:
                    self.xmax = self.frame_width
                if len(self.video_paths) <= 1:
                    subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                else:
                    print(f"{'Processing multiple videos or images'}")
                    # First check if the resolution of each video is consistent. If so, set the same subtitle area, otherwise set to None
                    global_size = None
                    for temp_video_path in self.video_paths:
                        temp_cap = cv2.VideoCapture(temp_video_path)
                        if global_size is None:
                            global_size = (int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        else:
                            temp_size = (int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            if temp_size != global_size:
                                print('not all video/images in same size, processing in full screen')
                                subtitle_area = None
                    else:
                        subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                y_p = self.ymin / self.frame_height
                h_p = (self.ymax - self.ymin) / self.frame_height
                x_p = self.xmin / self.frame_width
                w_p = (self.xmax - self.xmin) / self.frame_width
                self.set_subtitle_config(y_p, h_p, x_p, w_p)

                def task():
                    while self.video_paths:
                        video_path = self.video_paths.pop()
                        if subtitle_area is not None:
                            print(f"{'SubtitleArea'}：({self.ymin},{self.ymax},{self.xmin},{self.xmax})")
                        self.sr = backend.main.SubtitleRemover(video_path, subtitle_area, True)
                        self.__disable_button()
                        self.sr.run()
                Thread(target=task, daemon=True).start()
                self.video_cap.release()
                self.video_cap = None

    def _slide_event_handler(self, event, values):
        """
        When sliding the video progress bar/subtitle selection area slider:
        1) Check if the video exists, if so, display the corresponding video frame
        2) Draw rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event \
                == '-X-SLIDER-W-':
            # Determine if it is a single image
            if is_image_file(self.video_path):
                img = cv2.imread(self.video_path)
                self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - values['-Y-SLIDER-']))
                self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - values['-X-SLIDER-']))
                # Draw subtitle box
                y = int(values['-Y-SLIDER-'])
                h = int(values['-Y-SLIDER-H-'])
                x = int(values['-X-SLIDER-'])
                w = int(values['-X-SLIDER-W-'])
                self._update_preview(img, (y, h, x, w))
            elif self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(values['-SLIDER-'])
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if ret:
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height-values['-Y-SLIDER-']))
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width-values['-X-SLIDER-']))
                    # Draw subtitle box
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    self._update_preview(frame, (y, h, x, w))

    def _update_preview(self, frame, y_h_x_w):
        y, h, x, w = y_h_x_w
        # Draw subtitle box
        draw = cv2.rectangle(img=frame, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                             color=(0, 255, 0), thickness=3)
        # Resize video frame for player display
        resized_frame = self._img_resize(draw)
        # Display video frame
        self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())

    def _img_resize(self, image):
        top, bottom, left, right = (0, 0, 0, 0)
        height, width = image.shape[0], image.shape[1]
        # For images with unequal length and width, find the longest side
        longest_edge = height
        # Calculate how many pixels of width the short side needs to add to equal the long side
        if width < longest_edge:
            dw = longest_edge - width
            left = dw // 2
            right = dw - left
        else:
            pass
        # Add border to image
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(constant, (self.video_preview_width, self.video_preview_height))

    def set_subtitle_config(self, y, h, x, w):
        # Write to configuration file
        with open(self.subtitle_config_file, mode='w', encoding='utf-8') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y}\n')
            f.write(f'H = {h}\n')
            f.write(f'X = {x}\n')
            f.write(f'W = {w}\n')

    def parse_subtitle_config(self):
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        # If the configuration file does not exist, write to it
        if not os.path.exists(self.subtitle_config_file):
            self.set_subtitle_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        else:
            try:
                config = configparser.ConfigParser()
                config.read(self.subtitle_config_file, encoding='utf-8')
                conf_y_p, conf_h_p, conf_x_p, conf_w_p = float(config['AREA']['Y']), float(config['AREA']['H']), float(config['AREA']['X']), float(config['AREA']['W'])
                return conf_y_p, conf_h_p, conf_x_p, conf_w_p
            except Exception:
                self.set_subtitle_config(y_p, h_p, x_p, w_p)
                return y_p, h_p, x_p, w_p


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method("spawn")
        # Run GUI
        subtitleRemoverGUI = SubtitleRemoverGUI()
        subtitleRemoverGUI.run()
    except Exception as e:
        print(f'[{type(e)}] {e}')
        import traceback
        traceback.print_exc()
        msg = traceback.format_exc()
        err_log_path = os.path.join(os.path.expanduser('~'), 'VSR-Error-Message.log')
        with open(err_log_path, 'w', encoding='utf-8') as f:
            f.writelines(msg)
        import platform
        if platform.system() == 'Windows':
            os.system('pause')
        else:
            input()
