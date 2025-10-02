from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, BooleanProperty, ListProperty, NumericProperty
from kivy.clock import Clock
from kivy.animation import Animation
from plyer import filechooser
from ultralytics import YOLO
from PIL import Image
import pillow_heif
from kivy.core.window import Window
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
import os # Tambahkan import os untuk menghapus file

# Daftarkan HEIF agar bisa dibaca PIL
pillow_heif.register_heif_opener()

class ImageUploader(BoxLayout):
    # Properti yang akan menyimpan path gambar
    preview_image = StringProperty('')

    def select_image(self):
        # Logika untuk membuka dialog pemilihan file
        # Misalnya, menggunakan plyer.filechooser atau FileChooserIconView

        # Setelah pengguna memilih gambar:
        # file_path = "path/ke/gambar/yang/dipilih.jpg"
        # self.preview_image = file_path # Ini akan memicu pembaruan di KivyLang
        pass

class MainScreen(Screen):
    """Main screen dengan upload dan hasil deteksi"""
    image_path = StringProperty("")
    preview_image = StringProperty("")
    prediction_text = StringProperty("")
    confidence_text = StringProperty("")
    has_result = BooleanProperty(False)
    is_loading = BooleanProperty(False)
    show_error = BooleanProperty(False)
    error_message = StringProperty("")
    prognosis_color = ListProperty([0.12, 0.62, 0.33, 1])
    prognosis_bg_color = ListProperty([0.12, 0.62, 0.33, 0.1])

    # Properti untuk font size dinamis
    font_size_jumbo = NumericProperty(0)
    font_size_large = NumericProperty(0)
    font_size_medium = NumericProperty(0)
    font_size_normal = NumericProperty(0)
    font_size_small = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Bind fungsi on_window_size ke event on_resize dari Window
        Window.bind(on_resize=self.on_window_size)
        # Panggil sekali di awal untuk set ukuran font awal
        self.on_window_size(Window, Window.width, Window.height)
        self.model = None
        Clock.schedule_once(self.load_model)

    def on_window_size(self, window, width, height):
        """Dipanggil setiap kali ukuran window berubah."""
        # Base size bisa berdasarkan lebar atau tinggi, lebar biasanya lebih baik
        base_size = width

        # Atur ukuran font berdasarkan rasio dari lebar window
        # min() digunakan agar font tidak terlalu besar di layar desktop
        # max() digunakan agar font tidak terlalu kecil di layar mobile
        self.font_size_jumbo = max(18, min(base_size * 0.08, 32))
        self.font_size_large = max(16, min(base_size * 0.06, 24))
        self.font_size_medium = max(14, min(base_size * 0.05, 20))
        self.font_size_normal = max(12, min(base_size * 0.04, 16))
        self.font_size_small = max(10, min(base_size * 0.03, 14))

    def load_model(self, dt):
        """Memuat model AI."""
        try:
            # Ganti dengan path model Anda yang benar
            self.model = YOLO("best100.pt")
            print("Model loaded successfully.")
        except Exception as e:
            self.show_error_message(f"Model load failed: {e}")

    def select_image(self):
        """Membuka file chooser."""
        try:
            filechooser.open_file(on_selection=self.handle_selection, filters=[("Image files", "*.jpg", "*.jpeg", "*.png", "*.heic")])
        except Exception as e:
            self.show_error_message(f"Failed to open file chooser: {e}")

    def handle_selection(self, selection):
        """Menangani file yang dipilih."""
        if not selection:
            return

        self.image_path = selection[0]

        if self.image_path.lower().endswith('.heic'):
            self.convert_heic()
        else:
            self.preview_image = self.image_path
            self.show_success_feedback()

    def convert_heic(self):
        """Mengonversi file HEIC ke JPG."""
        try:
            heif_file = pillow_heif.read_heif(self.image_path)
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw").convert("RGB")

            # Simpan file yang dikonversi di direktori sementara
            temp_dir = App.get_running_app().user_data_dir
            temp_filename = os.path.basename(self.image_path).rsplit(".", 1)[0] + ".jpg"
            new_path = os.path.join(temp_dir, temp_filename)

            img.save(new_path, "JPEG")

            self.image_path = new_path
            self.preview_image = new_path
            self.show_success_feedback()
        except Exception as e:
            self.show_error_message(f"Failed to convert HEIC: {str(e)}")

    def show_success_feedback(self):
        """Menampilkan feedback setelah gambar berhasil dipilih."""
        self.show_error = False
        self.has_result = False
        if self.ids.get('preview_img'):
            self.ids.preview_img.reload()

    def show_error_message(self, message):
        """Menampilkan pesan error."""
        self.error_message = message
        self.show_error = True
        self.has_result = False

    def detect_image(self):
        """Memulai proses deteksi."""
        if not self.image_path:
            self.show_error_message("Please select an image first.")
            return
        if not self.model:
            self.show_error_message("AI Model is not loaded yet. Please wait.")
            return

        self.is_loading = True
        self.has_result = False
        self.show_error = False

        Clock.schedule_once(self.run_detection, 0.5)

    def run_detection(self, dt):
        """Menjalankan deteksi dan menampilkan hasil."""
        original_image_path = self.image_path # Simpan path asli untuk dihapus nanti
        
        try:
            results = self.model(original_image_path)

            if results and results[0].boxes:
                box = results[0].boxes[0]
                pred_class_idx = int(box.cls[0])
                pred_class_name = self.model.names[pred_class_idx]
                confidence = float(box.conf[0]) * 100

                self.prediction_text = pred_class_name
                self.confidence_text = f"{confidence:.2f}%"

                if "Normal" in pred_class_name:
                    self.prognosis_color = [0.12, 0.62, 0.33, 1]
                    self.prognosis_bg_color = [0.12, 0.62, 0.33, 0.1]
                else:
                    self.prognosis_color = [0.78, 0.16, 0.16, 1]
                    self.prognosis_bg_color = [0.78, 0.16, 0.16, 0.1]

                self.has_result = True
                self.scroll_to_results()
            else:
                self.show_error_message("No condition detected in the image.")

        except Exception as e:
            self.show_error_message(f"An error occurred during detection: {e}")

        finally:
            self.is_loading = False
            # [MODIFIKASI: HAPUS GAMBAR SETELAH HASIL MUNCUL]
            self.preview_image = ""
            self.image_path = ""
            # ---------------------------------------------------

    def scroll_to_results(self):
        """Scroll ke bagian hasil."""
        if self.ids.get('results_section'):
            scroll_view = self.ids.scroll_view
            results_widget = self.ids.results_section
            # Scroll ke bagian bawah di mana hasil berada
            scroll_view.scroll_to(results_widget, padding=10, animate=True)

class HelloWorld(App):
    """Main application"""
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    Window.size = (430, 832)
    HelloWorld().run()