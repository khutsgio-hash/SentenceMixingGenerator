import sys
import os
import re
import wave
import threading
import uuid  # Added for random filename generation
from PyQt5 import QtWidgets, QtGui, QtCore
import pyaudio
from g2p_en import G2p
import audioop
import random

class TextToSpeech:
    CHUNK = 1024
    PHONEME_MAPPING = {
        'AW': ['AE', 'OW'], 'DH': ['D'], 'EY': ['EH', 'IY'], 'JH': ['CH'],
        'SH': ['CH'], 'TH': ['D'], 'ZH': ['CH'], 'AE': ['AA'],
        'AO': ['AA', 'OW'], 'ER': ['AA'], 'IH': ['IY'],
        'OY': ['OW', 'Y', 'IY'], 'UH': ['UW']
    }

    def __init__(self, character_folder):
        self.character_folder = character_folder
        self.g2p = G2p()
        self.word_pause = 0.001
        self.fade_duration = 0.01 # Updated crossfade duration to 0.08 seconds

    def _is_vowel(self, phoneme):
        vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        return phoneme.strip().upper() in vowels

    def _pick_random_variant(self, base_path):
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        if not os.path.isdir(directory): return None
        pattern = re.compile(rf"^{re.escape(base_name)}(_\d+)?\.wav$", re.IGNORECASE)
        candidates = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
        return random.choice(candidates) if candidates else None

    def _get_phoneme_data(self, phoneme, target_ch, target_w, target_r):
        if phoneme == "AH0":
            data = self._get_phoneme_data("EH", target_ch, target_w, target_r)
            if data:
                frame_size = target_w * target_ch
                if len(data) > frame_size * 2: return data[frame_size:-frame_size]
            return data

        base = os.path.join(self.character_folder, f"{phoneme}.wav")
        path = self._pick_random_variant(base)
        if path: return self._normalize_wav(path, target_ch, target_w, target_r)
        
        if phoneme in self.PHONEME_MAPPING:
            combined_data = b""
            for sub_p in self.PHONEME_MAPPING[phoneme]:
                sub_data = self._get_phoneme_data(sub_p, target_ch, target_w, target_r)
                if sub_data:
                    if combined_data:
                        # Apply crossfading sequentially between sub-phonemes
                        faded_curr, leftover_next = self._apply_crossfade(combined_data, sub_data, target_r, target_w)
                        combined_data = faded_curr + leftover_next
                    else:
                        combined_data = sub_data
            return combined_data if combined_data else None
        return None

    def _normalize_wav(self, filepath, target_channels, target_width, target_rate):
        with wave.open(filepath, 'rb') as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            data = wf.readframes(wf.getnframes())
            if channels > 1: data = audioop.tomono(data, width, 0.5, 0.5)
            if width != target_width: data = audioop.lin2lin(data, width, target_width)
            if rate != target_rate: data, _ = audioop.ratecv(data, target_width, 1, rate, target_rate, None)
            return data

    def _apply_crossfade(self, current_data, next_data, rate, width):
        fade_size = int(rate * self.fade_duration) * width
        actual_fade = min(fade_size, len(current_data), len(next_data))
        if actual_fade <= 0: return current_data, next_data

        curr_main = current_data[:-actual_fade]
        ramp_out, ramp_in = bytearray(), bytearray()
        num_frames = actual_fade // width
        for i in range(num_frames):
            out_f, in_f = 1.0 - (i / num_frames), i / num_frames
            ramp_out += audioop.mul(current_data[len(curr_main) + (i*width) : len(curr_main) + ((i+1)*width)], width, out_f)
            ramp_in += audioop.mul(next_data[(i*width) : ((i+1)*width)], width, in_f)

        mixed_fade = audioop.add(bytes(ramp_out), bytes(ramp_in), width)
        return curr_main + mixed_fade, next_data[actual_fade:]

    def generate_audio_data(self, str_input):
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        raw_segments = []
        target_ch, target_w, target_r = 1, 2, 44100

        for token in tokens:
            if token in [".", "!", "?", ",", ";"]:
                dur = 0.5 if token in [".", "!", "?"] else 0.25
                raw_segments.append({"data": b'\x00' * int(target_r * dur * target_w * target_ch), "is_pause": True})
                continue

            word_wav = self._pick_random_variant(os.path.join(self.character_folder, "words", f"{token.upper()}.wav"))
            if word_wav:
                raw_segments.append({"data": self._normalize_wav(word_wav, target_ch, target_w, target_r), "is_pause": False})
            else:
                phonemes = self.g2p(token)
                valid_ps = [re.sub(r'\d+', '', p) if p != "AH0" else p for p in phonemes]
                valid_ps = [p for p in valid_ps if re.match(r'[A-Z]+[0-9]*', p)]
                if valid_ps and valid_ps[-1] in ["AH", "AE", "AH0"]: valid_ps[-1] = "AA"
                for p_clean in valid_ps:
                    data = self._get_phoneme_data(p_clean, target_ch, target_w, target_r)
                    if data: raw_segments.append({"data": data, "is_pause": False})

            raw_segments.append({"data": b'\x00' * int(target_r * self.word_pause * target_w * target_ch), "is_pause": True})

        final_audio = b""
        for i in range(len(raw_segments)):
            curr_data = raw_segments[i]["data"]
            if i + 1 < len(raw_segments) and not raw_segments[i]["is_pause"] and not raw_segments[i+1]["is_pause"]:
                faded_curr, leftover_next = self._apply_crossfade(curr_data, raw_segments[i+1]["data"], target_r, target_w)
                final_audio += faded_curr
                raw_segments[i+1]["data"] = leftover_next
            else:
                final_audio += curr_data
        return final_audio

    def render_to_file(self, str_input, output_path):
        audio_data = self.generate_audio_data(str_input)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
            wf.writeframes(audio_data)

    def get_pronunciation(self, str_input):
        temp_path = "temp_playback.wav"
        self.render_to_file(str_input, temp_path)
        def play():
            if not os.path.exists(temp_path): return
            wf = wave.open(temp_path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            data = wf.readframes(self.CHUNK)
            while data: stream.write(data); data = wf.readframes(self.CHUNK)
            stream.stop_stream(); stream.close(); p.terminate(); wf.close()
            try: os.remove(temp_path)
            except: pass
        threading.Thread(target=play).start()

class TTSGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentence Mixing Generator")
        self.setMinimumSize(600, 850)
        self.categories = {}
        self.load_categories()
        
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', Arial; }
            QTabWidget::pane { border: 1px solid #333; background: #121212; }
            QTabBar::tab { background: #1e1e1e; padding: 10px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #333; border-bottom: 2px solid #f1c40f; }
            QComboBox, QLineEdit, QPlainTextEdit { background-color: #1e1e1e; border: 1px solid #333; padding: 8px; border-radius: 4px; }
            #SynthesizeBtn { background-color: #f1c40f; color: #000; font-weight: bold; font-size: 16px; border-radius: 4px; padding: 12px; }
            #SynthesizeBtn:hover { background-color: #d4ac0d; }
            #SaveBtn { background-color: #2ecc71; color: #000; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 10px; }
            #SaveBtn:hover { background-color: #27ae60; }
            #BrowseBtn { background-color: #333; color: #fff; border-radius: 4px; padding: 5px 15px; }
            QLabel { font-weight: bold; color: #aaa; }
        """)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        
        self.batch_tab = QtWidgets.QWidget()
        self.setup_batch_tab()
        self.tabs.addTab(self.batch_tab, "TTS")
        
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)

    def setup_batch_tab(self):
        layout = QtWidgets.QVBoxLayout(self.batch_tab)
        layout.setSpacing(15)

        selector_layout = QtWidgets.QGridLayout()
        self.category_selector = QtWidgets.QComboBox()
        self.category_selector.addItems(self.categories.keys())
        self.category_selector.currentTextChanged.connect(self.update_character_list)
        
        self.character_dropdown = QtWidgets.QComboBox()
        self.character_dropdown.currentTextChanged.connect(self.update_preview_image)
        
        selector_layout.addWidget(QtWidgets.QLabel("VOICE CATEGORY"), 0, 0)
        selector_layout.addWidget(self.category_selector, 1, 0)
        selector_layout.addWidget(QtWidgets.QLabel("CHARACTER"), 0, 1)
        selector_layout.addWidget(self.character_dropdown, 1, 1)
        layout.addLayout(selector_layout)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(200, 200)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 2px solid #333; border-radius: 100px; background: #000;")
        layout.addWidget(self.preview_label, alignment=QtCore.Qt.AlignCenter)

        layout.addWidget(QtWidgets.QLabel("SAVE DESTINATION FOLDER"))
        path_layout = QtWidgets.QHBoxLayout()
        self.path_input = QtWidgets.QLineEdit()
        self.path_input.setPlaceholderText("Select folder to save audio...")
        self.btn_browse = QtWidgets.QPushButton("Browse")
        self.btn_browse.setObjectName("BrowseBtn")
        self.btn_browse.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)

        layout.addWidget(QtWidgets.QLabel("TEXT TO SYNTHESIZE"))
        self.batch_text = QtWidgets.QPlainTextEdit()
        layout.addWidget(self.batch_text)

        btn_layout = QtWidgets.QHBoxLayout()
        
        btn_speak = QtWidgets.QPushButton("SPEAK")
        btn_speak.setObjectName("SynthesizeBtn")
        btn_speak.clicked.connect(self.speak_batch)
        
        btn_save = QtWidgets.QPushButton("BATCH SAVE (.WAV)")
        btn_save.setObjectName("SaveBtn")
        btn_save.clicked.connect(self.save_batch)
        
        btn_layout.addWidget(btn_speak, 2)
        btn_layout.addWidget(btn_save, 1)
        layout.addLayout(btn_layout)

        if self.categories: self.update_character_list(self.category_selector.currentText())

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.path_input.setText(folder)

    def update_character_list(self, cat):
        self.character_dropdown.clear()
        chars = list(self.categories.get(cat, {}).keys())
        self.character_dropdown.addItems(chars)
        if chars: self.update_preview_image(chars[0])

    def update_preview_image(self, char_name):
        cat = self.category_selector.currentText()
        if not char_name or cat not in self.categories: return
        
        path = self.categories[cat][char_name]
        icon_p = os.path.join(path, 'profile.png')
        
        if os.path.exists(icon_p):
            pixmap = QtGui.QPixmap(icon_p).scaled(200, 200, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            size = pixmap.size()
            mask = QtGui.QBitmap(size)
            mask.fill(QtCore.Qt.color0)
            painter = QtGui.QPainter(mask)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtCore.Qt.color1)
            painter.drawEllipse(0, 0, size.width(), size.height())
            painter.end()
            pixmap.setMask(mask)
            self.preview_label.setPixmap(pixmap)
        else:
            self.preview_label.setText("No Preview")

    def load_categories(self):
        base_path = 'assets/characters'
        if not os.path.isdir(base_path): return
        for cat in os.listdir(base_path):
            cat_p = os.path.join(base_path, cat)
            if os.path.isdir(cat_p):
                self.categories[cat] = {char: os.path.join(cat_p, char) for char in os.listdir(cat_p) if os.path.isdir(os.path.join(cat_p, char))}

    def get_current_tts_engine(self):
        char_name = self.character_dropdown.currentText()
        if not char_name: return None
        path = self.categories[self.category_selector.currentText()][char_name]
        return TextToSpeech(path)

    def speak_batch(self):
        engine = self.get_current_tts_engine()
        text = self.batch_text.toPlainText().strip()
        if engine and text:
            engine.get_pronunciation(text)

    def save_batch(self):
        engine = self.get_current_tts_engine()
        text = self.batch_text.toPlainText().strip()
        folder = self.path_input.text().strip()

        if not engine:
            QtWidgets.QMessageBox.warning(self, "Error", "No character selected.")
            return
        if not text:
            QtWidgets.QMessageBox.warning(self, "Error", "No text provided.")
            return
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a valid destination folder.")
            return

        try:
            # Generates a unique random filename (e.g. tts_a1b2c3d4.wav)
            random_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_file = os.path.join(folder, random_filename)
            
            engine.render_to_file(text, output_file)
            QtWidgets.QMessageBox.information(self, "Success", f"Audio saved as:\n{random_filename}\n\nIn folder:\n{folder}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save audio: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Arial", 12))
    gui = TTSGui()
    gui.show()
    sys.exit(app.exec_())