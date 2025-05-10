import cv2
import numpy as np
import os
from glob import glob
from statistics import mean

def preprocess_and_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    return image[y:y+h, x:x+w]

def auto_detect_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    vertical = binary.copy()
    horizontal = binary.copy()

    vertical_size = vertical.shape[0] // 50
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    horizontal_size = horizontal.shape[1] // 30
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            cells.append((x, y, w, h))

    return cells

def estimate_type_by_cell_shape(cells, image, matrix, resize_dim=(64, 64)):
    heights = []
    widths = []
    contour_counts = []

    for col in range(25):
        for row in range(50):
            cell_data = matrix[col][row]
            if cell_data is None:
                continue
            x, y, w, h = cell_data
            cell = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            heights.append(h)
            widths.append(w)
            contour_counts.append(len(contours))

    avg_h = mean(heights)
    avg_w = mean(widths)
    avg_contours = mean(contour_counts)
    ratio = avg_h / avg_w if avg_w != 0 else 1

    if avg_h > 80:
        if avg_contours > 10:
            return "velika_pisana"
        else:
            return "velika_tiskana"
    elif avg_h > 40:
        if avg_contours > 10:
            return "mala_pisana"
        else:
            return "mala_tiskana"
    return "neznana"

def normaliziraj_znak(znak):
    posebni = {"Č": "CC", "Š": "SS", "Ž": "ZZ"}
    return posebni.get(znak, znak)  # če ni poseben, vrne original

def process_first_5_images_auto_type(input_dir, output_dir_root, resize_dim=(64, 64)):
    os.makedirs(output_dir_root, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))[:2]

    abeceda = list("ABCČDEFGHIJKLMNOPRSŠTUVZŽ")
    cols = 25
    rows = 50

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Napaka pri branju slike: {filename}")
            continue

        cropped = preprocess_and_crop(image)
        cells = auto_detect_cells(cropped)

        if len(cells) != rows * cols:
            print(f"Opozorilo: pričakovanih {rows * cols} celic, najdenih {len(cells)} v {filename}")
            print("Nadaljujem z delnim izrezom...")

        # Najprej ustvarimo matriko
        cells_sorted = sorted(cells, key=lambda r: (r[0], r[1]))  # sort by x (stolpec), y (vrstica)
        matrix = np.full((cols, rows), None)

        for i in range(min(rows * cols, len(cells_sorted))):
            c = i // rows
            r = i % rows
            if c < cols and r < rows:
                matrix[c][r] = cells_sorted[i]

        # Šele zdaj pokličemo funkcijo za oceno tipa črk
        tip = estimate_type_by_cell_shape(cells, cropped, matrix, resize_dim)

        # Izhodna mapa glede na prepoznani tip
        output_dir = os.path.join(output_dir_root, tip)
        os.makedirs(output_dir, exist_ok=True)

        # Shrani po črkah (stolpec za stolpcem)
        for col in range(cols):
            letter = abeceda[col]
            for row in range(rows):
                cell_data = matrix[col][row]
                if cell_data is None:
                    continue
                x, y, w, h = cell_data
                cell = cropped[y:y+h, x:x+w]
                resized = cv2.resize(cell, resize_dim)
                base = os.path.splitext(filename)[0]
                znak = normaliziraj_znak(letter)
                out_name = f"{base}_{tip}_{znak}_{row + 1:02d}.png"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, resized)

        print(f"{filename}: zaznano kot {tip}, izrezanih približno {min(rows * cols, len(cells))} črk")

if __name__ == "__main__":
    process_first_5_images_auto_type("abeceda", "izhod_vse_auto")
