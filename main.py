import cv2
import csv
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

def normalize_character(char):
    special = {"Č": "CC", "Š": "SS", "Ž": "ZZ"}
    return special.get(char, char)  # če ni poseben, vrne original

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
                znak = normalize_character(letter)
                out_name = f"{base}_{tip}_{znak}_{row + 1:02d}.png"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, resized)

        print(f"{filename}: zaznano kot {tip}, izrezanih približno {min(rows * cols, len(cells))} črk")

def process_all_from_subfolders(input_root, output_root, resize_dim=(64, 64)):
    os.makedirs(output_root, exist_ok=True)
    abeceda = list("ABCČDEFGHIJKLMNOPRSŠTUVZŽ")
    cols = 25
    rows = 50

    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        image_paths = sorted(glob(os.path.join(subfolder_path, "*.png")))
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Napaka pri branju slike: {filename}")
                continue

            cropped = preprocess_and_crop(image)
            cells = auto_detect_cells(cropped)

            cells_sorted = sorted(cells, key=lambda r: (r[0], r[1]))
            matrix = np.full((cols, rows), None)

            for i in range(min(rows * cols, len(cells_sorted))):
                c = i // rows
                r = i % rows
                if c < cols and r < rows:
                    matrix[c][r] = cells_sorted[i]

            tip = subfolder  # Uporabi kar ime mape (npr. 'mala tiskana')
            output_dir = os.path.join(output_root, tip.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)

            for col in range(cols):
                letter = abeceda[col]
                znak = normalize_character(letter)
                for row in range(rows):
                    cell_data = matrix[col][row]
                    if cell_data is None:
                        continue
                    x, y, w, h = cell_data
                    cell = cropped[y:y+h, x:x+w]
                    resized = cv2.resize(cell, resize_dim)
                    base = os.path.splitext(filename)[0]
                    out_name = f"{base}_{tip.replace(' ', '_')}_{znak}_{row + 1:02d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    cv2.imwrite(out_path, resized)

            print(f"{filename} ({tip}) → izrezanih približno {min(rows * cols, len(cells))} črk")

def delete_images_ending_with_01(root_folder):
    count = 0
    for subdir, _, _ in os.walk(root_folder):
        for file_path in glob(os.path.join(subdir, "*.png")):
            if file_path.endswith("_01.png"):
                try:
                    os.remove(file_path)
                    print(f"Zbrisano: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Napaka pri brisanju {file_path}: {e}")
    print(f"\nSkupno izbrisanih slik: {count}")

def extract_features(image, split=4):
    h, w = image.shape
    features = []

    block_h = h // split
    block_w = w // split

    for i in range(split):
        for j in range(split):
            y_start = i * block_h
            x_start = j * block_w
            block = image[y_start:y_start + block_h, x_start:x_start + block_w]

            total_pixels = block.size
            dark_pixels = np.sum(block < 128)
            dark_ratio = dark_pixels / total_pixels

            features.append(dark_pixels)
            features.append(round(dark_ratio, 4))  # zaokroženo

    return features

def parse_filename(filename):
    #('sc032', 'mala_pisana', 'A', '02')
    base = os.path.splitext(os.path.basename(filename))[0]
    deli = base.split("_")
    if len(deli) >= 5:
        return deli[0], f"{deli[1]}_{deli[2]}", deli[3], deli[4]
    return base, "neznano", "?", "00"

def extract_features_from_images(image_dir, grid_size=4, output_csv="znacilke.csv"):
    header = ["ime_slike", "tip_crke", "crka", "stevilka"]

    for i in range(grid_size * grid_size):
        header.append(f"temni_{i}")
        header.append(f"razmerje_{i}")

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for subfolder, _, _ in os.walk(image_dir):
            for image_path in glob(os.path.join(subfolder, "*.png")):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, (64, 64))  # enotna velikost
                _, binary_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)

                ime_slike, tip_crke, crka, stevilka = parse_filename(image_path)
                znacilke = extract_features(binary_image, grid_size)
                vrstica = [ime_slike, tip_crke, crka, stevilka] + znacilke
                writer.writerow(vrstica)

    print(f"Zapisano v datoteko: {output_csv}")

if __name__ == "__main__":
    nameAlphabet = "izhod_abeceda"
    #process_first_5_images_auto_type("test crke", "izhod_vse_auto")
    #process_all_from_subfolders("new abeceda", nameAlphabet)
    #delete_images_ending_with_01(nameAlphabet)
    extract_features_from_images("izhod_abeceda", grid_size=4, output_csv="znacilke.csv")


