import cv2
import csv
import numpy as np
import os
from glob import glob
from statistics import mean

def preprocess_and_crop(image):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # pixels brighter than 240 become black (0), others become white (255)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find external contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original image
    if not contours:
        return image

    # Identify the largest contour by area (character grid/table)
    biggest = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(biggest)

    # Crop the original image to this bounding box and return it
    return image[y:y+h, x:x+w]

def auto_detect_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    blur = cv2.GaussianBlur(gray, (3, 3), 0)   # Apply Gaussian blur to reduce noise
    # Apply binary inverse thresholding to emphasize the grid
    _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    # Create separate copies to detect vertical and horizontal line
    vertical = binary.copy()
    horizontal = binary.copy()

    # --- Detect vertical lines ---
    vertical_size = vertical.shape[0] // 50
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    # --- Detect horizontal lines ---
    horizontal_size = horizontal.shape[1] // 30
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    # Combine vertical and horizontal lines to reconstruct the grid
    grid = cv2.add(horizontal, vertical)

    # Find all contours in the grid image
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # Filter out small noise contours
            cells.append((x, y, w, h))

    return cells

def estimate_type_by_cell_shape(image, matrix):
    heights = []
    widths = []
    contour_counts = []

    # Iterate through each cell in the 25x50 matrix
    for col in range(25):
        for row in range(50):
            cell_data = matrix[col][row]
            if cell_data is None:
                continue
            x, y, w, h = cell_data
            cell = image[y:y+h, x:x+w]

            # Convert each cell to grayscale and apply binary inverse thresholding
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            # Count the number of external contours (strokes or curves in the letter)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Save height, width, and number of contours for analysis
            heights.append(h)
            widths.append(w)
            contour_counts.append(len(contours))

    # Calculate averages
    avg_h = mean(heights)
    avg_w = mean(widths)
    avg_contours = mean(contour_counts)
    ratio = avg_h / avg_w if avg_w != 0 else 1

    # Estimate type of writing based on size and complexity
    if avg_h > 80:
        if avg_contours > 10 or ratio > 1.3:
            return "velika_pisana"
        else:
            return "velika_tiskana"
    elif avg_h > 40:
        if avg_contours > 10 or ratio > 1.3:
            return "mala_pisana"
        else:
            return "mala_tiskana"
    else:
        return "neznana"

def normalize_character(char):
    special = {"Č": "CC", "Š": "SS", "Ž": "ZZ"} # Define replacements for special Slovenian characters
    return special.get(char, char)  # otherwise return the character unchanged

def process_first_5_images_auto_type(input_dir, output_dir_root, resize_dim=(64, 64)):
    os.makedirs(output_dir_root, exist_ok=True)

    # Get the first 2 PNG images in the input directory (for testing)
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))[:2]

    abeceda = list("ABCČDEFGHIJKLMNOPRSŠTUVZŽ")
    cols = 25 # number of columns = number of letters
    rows = 50  # number of samples per letter

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Napaka pri branju slike: {filename}")
            continue

        # Crop the image to remove empty space and focus on the table
        cropped = preprocess_and_crop(image)
        # Detect all cell regions in the image
        cells = auto_detect_cells(cropped)

        # Warn if the number of detected cells is not as expected
        if len(cells) != rows * cols:
            print(f"Opozorilo: pričakovanih {rows * cols} celic, najdenih {len(cells)} v {filename}")

        # Create a 2D matrix [columns][rows] to store cell coordinates
        cells_sorted = sorted(cells, key=lambda r: (r[0], r[1]))  # sort by x (stolpec), y (vrstica)
        matrix = np.full((cols, rows), None)

        # Fill the 2D matrix [columns][rows] with detected cell positions
        for i in range(min(rows * cols, len(cells_sorted))):
            c = i // rows # column index
            r = i % rows # row index
            if c < cols and r < rows:
                matrix[c][r] = cells_sorted[i]

        # Estimate writing type (e.g., printed/cursive, upper/lower)
        tip = estimate_type_by_cell_shape(cells, cropped)

        # Create output directory for the detected type
        output_dir = os.path.join(output_dir_root, tip)
        os.makedirs(output_dir, exist_ok=True)

        # Save each cell as an individual image, grouped by letter
        for col in range(cols):
            letter = abeceda[col]
            for row in range(rows):
                cell_data = matrix[col][row]
                if cell_data is None:
                    continue  # skip empty cells (not detected)

                x, y, w, h = cell_data
                cell = cropped[y:y+h, x:x+w]  # crop the cell region from the image
                resized = cv2.resize(cell, resize_dim) # resize to standard dimensions
                base = os.path.splitext(filename)[0] # remove file extension
                znak = normalize_character(letter) # normalize special characters (Č → CC, etc.)
                out_name = f"{base}_{tip}_{znak}_{row + 1:02d}.png" # construct output filename
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, resized)

        # save the character image to the output folder
        print(f"{filename}: zaznano kot {tip}, izrezanih približno {min(rows * cols, len(cells))} črk")

def process_all_from_subfolders(input_root, output_root, resize_dim=(64, 64)):
    os.makedirs(output_root, exist_ok=True)
    abeceda = list("ABCČDEFGHIJKLMNOPRSŠTUVZŽ")
    cols = 25  # number of letter columns
    rows = 50  # number of letter samples per column

    # Iterate over all subfolders (each representing a letter type)
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue # skip if it's not a folder

        # Get all PNG images from the current subfolder
        image_paths = sorted(glob(os.path.join(subfolder_path, "*.png")))
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Napaka pri branju slike: {filename}")
                continue

            cropped = preprocess_and_crop(image)  # Crop the table region from the image
            cells = auto_detect_cells(cropped)  # Detect all character cells in the image

            # Sort cells by column (x) and then by row (y)
            cells_sorted = sorted(cells, key=lambda r: (r[0], r[1]))
            # Create an empty 2D matrix [columns][rows] for storing cell positions
            matrix = np.full((cols, rows), None)

            for i in range(min(rows * cols, len(cells_sorted))):
                c = i // rows
                r = i % rows
                if c < cols and r < rows:
                    matrix[c][r] = cells_sorted[i]

            tip = subfolder   # Use the subfolder name as the type of writing (e.g., 'mala tiskana')
            output_dir = os.path.join(output_root, tip.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)

            # Iterate over each column (letter) and row (repetition)
            for col in range(cols):
                letter = abeceda[col]
                znak = normalize_character(letter) # replace special letters
                for row in range(rows):
                    cell_data = matrix[col][row]
                    if cell_data is None:
                        continue
                    x, y, w, h = cell_data
                    cell = cropped[y:y+h, x:x+w]
                    resized = cv2.resize(cell, resize_dim)

                    # Construct the output filename
                    base = os.path.splitext(filename)[0]
                    out_name = f"{base}_{tip.replace(' ', '_')}_{znak}_{row + 1:02d}.png"
                    out_path = os.path.join(output_dir, out_name)

                    # Save the character image
                    cv2.imwrite(out_path, resized)

            print(f"{filename} ({tip}) → izrezanih približno {min(rows * cols, len(cells))} črk")

def delete_images_ending_with_01(root_folder):
    count = 0
    # Walk through all subdirectories and files in the root folder
    for subdir, _, _ in os.walk(root_folder):
        for file_path in glob(os.path.join(subdir, "*.png")):  # Get all .png files in the current subdirectory
            if file_path.endswith("_01.png"):    # Check if the file ends with '_01.png'
                try:
                    os.remove(file_path)  # Delete the file
                    print(f"Zbrisano: {file_path}")
                    count += 1
                except Exception as e:   # Handle potential deletion errors
                    print(f"Napaka pri brisanju {file_path}: {e}")
    print(f"\nSkupno izbrisanih slik: {count}")

def extract_features(image, split=4):
    # Get image dimensions
    h, w = image.shape
    features = []

    # Calculate block size based on the number of splits
    block_h = h // split
    block_w = w // split

    # Loop through each block in the grid
    for i in range(split):
        for j in range(split):
            # Calculate starting coordinates for the current block
            y_start = i * block_h
            x_start = j * block_w

            # Extract the block (subregion of the image)
            block = image[y_start:y_start + block_h, x_start:x_start + block_w]

            # Calculate number of pixels and number of dark pixels in the block
            total_pixels = block.size
            dark_pixels = np.sum(block < 128)

            # Compute the ratio of dark pixels
            dark_ratio = dark_pixels / total_pixels

            # Append both features: count and ratio (rounded)
            features.append(dark_pixels)
            features.append(round(dark_ratio, 4))

    return features

def parse_filename(filename):
    #('sc032', 'mala_pisana', 'A', '02')
    base = os.path.splitext(os.path.basename(filename))[0] # Remove path and extension to get base name
    deli = base.split("_")  # Split the base name by underscores

    # If the name is correctly structured (at least 5 parts), extract relevant metadata
    if len(deli) >= 5:
        return deli[0], f"{deli[1]}_{deli[2]}", deli[3], deli[4]
    return base, "neznano", "?", "00"

def extract_features_from_images(image_dir, grid_size=4, output_csv="znacilke.csv"):
    # Create CSV header: metadata + features
    header = ["ime_slike", "tip_crke", "crka", "stevilka"]  # metadata columns

    # Add feature names for each grid cell (dark pixel count + ratio)
    for i in range(grid_size * grid_size):
        header.append(f"temni_{i}") # number of dark pixels
        header.append(f"razmerje_{i}") # ratio of dark pixels

    # Open the output CSV file for writing
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Go through all subdirectories and their PNG images
        for subfolder, _, _ in os.walk(image_dir):
            for image_path in glob(os.path.join(subfolder, "*.png")):
                # Read image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, (64, 64)) # Resize image to standard size

                # Convert image to binary (white background, black letters)
                _, binary_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)

                # Extract metadata from filename
                ime_slike, tip_crke, crka, stevilka = parse_filename(image_path)
                # Extract grid-based features from the binary image
                znacilke = extract_features(binary_image, grid_size)
                # Compose the full row and write to CSV
                vrstica = [ime_slike, tip_crke, crka, stevilka] + znacilke
                writer.writerow(vrstica)

    print(f"Zapisano v datoteko: {output_csv}")

if __name__ == "__main__":
    nameAlphabet = "izhod_abeceda"
    #process_first_5_images_auto_type("test crke", "izhod_vse_auto")
    #process_all_from_subfolders("new abeceda", nameAlphabet)
    #delete_images_ending_with_01(nameAlphabet)
    extract_features_from_images("izhod_abeceda", grid_size=4, output_csv="znacilke.csv")


