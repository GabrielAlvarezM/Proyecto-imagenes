import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
from tqdm import tqdm  # Importamos tqdm para la barra de progreso
import json  # Importamos json a nivel global

# Configuracion para el procesamiento de imagenes.
class ImageProcessorConfig:
    def __init__(self, median_blur_ksize=5, adaptive_thresh_blocksize=11, adaptive_thresh_C=2, defect_area_threshold=100):
        self.median_blur_ksize = median_blur_ksize
        self.adaptive_thresh_blocksize = adaptive_thresh_blocksize
        self.adaptive_thresh_C = adaptive_thresh_C
        self.defect_area_threshold = defect_area_threshold

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig, image_paths: Optional[List[str]] = None):
        # Inicializa el procesador de imagenes con una configuracion y una lista de rutas de imagenes.
        self.config = config
        self.image_paths = image_paths or []
        self.images = []
        self.processed_images = []
        self.logger = self._setup_logger()
        self.logger.info("ImageProcessor inicializado.")

    def _setup_logger(self):
        logger = logging.getLogger('ImageProcessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(handler)
        return logger

    def load_images(self):
        # Carga imagenes en escala de grises desde las rutas proporcionadas.
        self.images = []
        for path in self.image_paths:
            if not os.path.isfile(path):
                self.logger.warning(f"Ruta no valida: {path}")
                continue
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is None:
                self.logger.warning(f"No se pudo cargar la imagen: {path}")
            else:
                self.images.append({'path': path, 'image': image})
                self.logger.info(f"Imagen cargada correctamente: {path}")
        if not self.images:
            self.logger.error("No se cargaron imagenes validas.")

    def add_image_path(self, path: str):
        # Anade una ruta de imagen a la lista de imagenes a procesar.
        if os.path.isfile(path):
            if path not in self.image_paths:
                self.image_paths.append(path)
                self.logger.info(f"Anadida ruta de imagen: {path}")
            else:
                self.logger.info(f"La ruta ya existe en la lista: {path}")
        else:
            self.logger.warning(f"Ruta no valida: {path}")

    def add_images_from_directory(self, directory: str, extensions: Optional[List[str]] = None):
        # Anade todas las imagenes de un directorio a la lista de imagenes a procesar.
        if not os.path.isdir(directory):
            self.logger.warning(f"Directorio no valido: {directory}")
            return
        extensions = extensions or ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    self.add_image_path(full_path)

    def apply_spatial_filter(self, image: np.ndarray) -> np.ndarray:
        # Aplica un filtro espacial (filtro de mediana) a la imagen.
        return cv.medianBlur(image, self.config.median_blur_ksize)

    def apply_equalization(self, image: np.ndarray) -> np.ndarray:
        # Aplica ecualizacion de histograma a la imagen.
        return cv.equalizeHist(image)

    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        # Aplica umbralizacion adaptativa a la imagen.
        return cv.adaptiveThreshold(
            image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            self.config.adaptive_thresh_blocksize,
            self.config.adaptive_thresh_C
        )

    def detect_defects(self, image: np.ndarray) -> List[np.ndarray]:
        # Detecta defectos en la imagen procesada.
        contours, _ = cv.findContours(
            image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        defects = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.config.defect_area_threshold:
                defects.append(contour)
        return defects

    def process_single_image(self, image_info: dict) -> dict:
        # Procesa una sola imagen y devuelve los resultados.
        path = image_info['path']
        image = image_info['image']
        try:
            filtered = self.apply_spatial_filter(image)
            equalized = self.apply_equalization(filtered)
            thresholded = self.apply_thresholding(equalized)
            defects = self.detect_defects(thresholded)
            self.logger.info(f"Procesamiento completo para: {path}")
            return {
                'path': path,
                'original': image,
                'processed': thresholded,
                'defects': defects
            }
        except Exception as e:
            self.logger.error(f"Error al procesar la imagen {path}: {e}")
            return {
                'path': path,
                'original': image,
                'processed': None,
                'defects': []
            }

    def process_images(self):
        # Procesa todas las imagenes cargadas utilizando hilos para mejorar el rendimiento.
        self.processed_images = []
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_single_image, self.images), total=len(self.images), desc='Procesando imagenes'))
            self.processed_images = results

    def save_images(self, output_dir: str):
        # Guarda las imagenes procesadas en el directorio especificado.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Directorio creado: {output_dir}")
        for idx, data in enumerate(self.processed_images):
            if data['processed'] is not None:
                filename = os.path.basename(data['path'])
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_processed{ext}")
                success = cv.imwrite(output_path, data['processed'])
                if success:
                    self.logger.info(f"Imagen procesada guardada: {output_path}")
                else:
                    self.logger.error(f"Error al guardar la imagen procesada: {output_path}")

    def generate_defect_report(self) -> List[dict]:
        # Genera un informe de defectos detectados en cada imagen.
        report = []
        for data in self.processed_images:
            num_defects = len(data['defects'])
            report.append({
                'image_path': data['path'],
                'num_defects': num_defects,
                'defects': [cv.boundingRect(d) for d in data['defects']]  # Agregamos informacion de los defectos
            })
            self.logger.info(f"Imagen: {data['path']}, Defectos detectados: {num_defects}")
        return report

    def save_defect_report(self, output_file: str):
        # Guarda el informe de defectos en un archivo JSON.
        report = self.generate_defect_report()
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Directorio creado para el informe: {output_dir}")
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
            self.logger.info(f"Informe de defectos guardado en: {output_file}")
        except Exception as e:
            self.logger.error(f"Error al guardar el informe de defectos: {e}")

    def display_images(self):
        # Muestra las imagenes originales y procesadas, incluyendo defectos detectados.
        for data in self.processed_images:
            original = data['original']
            processed = data['processed']
            defects = data['defects']
            if processed is None:
                continue

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Imagen Original')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(processed, cmap='gray')
            plt.title('Imagen Procesada')
            plt.axis('off')

            defect_image = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
            cv.drawContours(defect_image, defects, -1, (0, 0, 255), 1)
            plt.subplot(1, 3, 3)
            plt.imshow(defect_image)
            plt.title('Defectos Detectados')
            plt.axis('off')

            plt.show()

    def clear_data(self):
        # Limpia las listas de imagenes cargadas y procesadas.
        self.images = []
        self.processed_images = []
        self.logger.info("Datos de imagenes limpiados.")

    def set_config(self, config: ImageProcessorConfig):
        # Actualiza la configuracion del procesador de imagenes.
        self.config = config
        self.logger.info("Configuracion actualizada.")

# Ejemplo de uso:

if __name__ == "__main__":
    # Configuracion personalizada
    config = ImageProcessorConfig(
        median_blur_ksize=5,
        adaptive_thresh_blocksize=11,
        adaptive_thresh_C=2,
        defect_area_threshold=100
    )

    processor = ImageProcessor(config=config)
    # Anadir rutas de imagenes predeterminadas
    # processor.add_image_path("ruta/a/imagen1.png")
    # processor.add_image_path("ruta/a/imagen2.png")
    # processor.add_image_path("ruta/a/imagen3.png")
    # processor.add_image_path("ruta/a/imagen4.png")

    # O anadir imagenes desde un directorio
    # processor.add_images_from_directory("ruta/al/directorio/de/imagenes")

    # Cargar imagenes
    # processor.load_images()

    # Procesar imagenes
    # processor.process_images()

    # Guardar imagenes procesadas
    # processor.save_images("directorio_de_salida")

    # Generar informe de defectos y guardarlo
    # processor.save_defect_report("informe_defectos.json")

    # Mostrar imagenes
    # processor.display_images()
