"""
detect_deforestation.py — Detect deforestation from satellite imagery.

This script implements the change detection pipeline:
  1. Splits satellite images into patches
  2. Classifies each patch using the trained model
  3. Compares land cover maps from two time periods
  4. Flags patches where Forest → non-forest (deforestation)
  5. Generates visualizations and summary reports

Usage:
  # Demo mode (uses EuroSAT images to simulate the pipeline):
  python detect_deforestation.py --demo

  # Real satellite imagery mode:
  python detect_deforestation.py --before path/to/image_2018.tif --after path/to/image_2024.tif

  # Custom patch size:
  python detect_deforestation.py --demo --patch-size 128
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "satellite_classifier.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data", "EuroSAT", "2750")

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Classes that indicate deforestation when Forest transitions to them
DEFORESTATION_TARGETS = {'AnnualCrop', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential'}

# Color map for visualization
CLASS_COLORS = {
    'AnnualCrop':            (255, 255, 102),   # Yellow
    'Forest':                (34, 139, 34),      # Forest green
    'HerbaceousVegetation':  (144, 238, 144),   # Light green
    'Highway':               (128, 128, 128),    # Gray
    'Industrial':            (178, 102, 255),    # Purple
    'Pasture':               (102, 204, 0),      # Lime
    'PermanentCrop':         (204, 153, 0),      # Orange
    'Residential':           (255, 0, 0),        # Red
    'River':                 (0, 102, 204),       # Blue
    'SeaLake':               (0, 0, 180),         # Dark blue
}

# ----------------------------
# Transform for inference
# ----------------------------

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# Core Functions
# ----------------------------

def load_model():
    """Load the trained satellite classifier."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No saved model found at {MODEL_PATH}")
        print("Please run train.py first.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded (val_accuracy={checkpoint['val_accuracy']:.4f})")
    return model


def split_image_into_patches(image_path, patch_size=64):
    """
    Divide a satellite image into a grid of patches.

    Args:
        image_path: Path to the satellite image
        patch_size: Size of each square patch in pixels (default 64)

    Returns:
        patches: List of (PIL.Image, x_pos, y_pos) tuples
        grid_shape: (num_cols, num_rows) of the patch grid
        original_image: The original PIL image
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    patches = []
    num_cols = width // patch_size
    num_rows = height // patch_size

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * patch_size
            y = row * patch_size
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, x, y))

    print(f"  Image: {width}x{height} -> {num_cols}x{num_rows} grid = {len(patches)} patches ({patch_size}x{patch_size})")
    return patches, (num_cols, num_rows), img


def classify_patches(model, patches, batch_size=64):
    """
    Classify each patch using the trained model.

    Args:
        model: The trained classifier
        patches: List of (PIL.Image, x, y) tuples
        batch_size: Number of patches to process at once

    Returns:
        results: List of (class_name, class_idx, confidence, x, y) tuples
    """
    results = []

    # Process in batches for efficiency
    patch_images = [p[0] for p in patches]
    positions = [(p[1], p[2]) for p in patches]

    for i in tqdm(range(0, len(patch_images), batch_size), desc="  Classifying", unit="batch"):
        batch_imgs = patch_images[i:i + batch_size]
        batch_pos = positions[i:i + batch_size]

        # Transform and stack into a batch tensor
        batch_tensors = torch.stack([inference_transform(img) for img in batch_imgs])
        batch_tensors = batch_tensors.to(DEVICE)

        with torch.no_grad():
            outputs = model(batch_tensors)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

        for j in range(len(batch_imgs)):
            class_idx = predicted[j].item()
            confidence = confidences[j].item()
            x, y = batch_pos[j]
            results.append((CLASS_NAMES[class_idx], class_idx, confidence, x, y))

    return results


def build_land_cover_map(results, grid_shape, patch_size=64):
    """
    Build a 2D land cover map from classification results.

    Args:
        results: List of (class_name, class_idx, confidence, x, y) tuples
        grid_shape: (num_cols, num_rows) of the patch grid
        patch_size: Size of each patch in pixels (must match the value used in
                    split_image_into_patches; default 64)

    Returns:
        land_cover_map: 2D numpy array of class names, shaped (rows, cols)
        confidence_map: 2D numpy array of confidence scores
    """
    num_cols, num_rows = grid_shape
    land_cover_map = np.empty((num_rows, num_cols), dtype=object)
    confidence_map = np.zeros((num_rows, num_cols))

    for class_name, class_idx, confidence, x, y in results:
        col = x // patch_size
        row = y // patch_size
        if row < num_rows and col < num_cols:
            land_cover_map[row, col] = class_name
            confidence_map[row, col] = confidence

    return land_cover_map, confidence_map


def detect_changes(map_before, map_after):
    """
    Compare two land cover maps and detect deforestation events.

    A deforestation event = a patch classified as "Forest" in the before map
    that changed to a non-forest class (AnnualCrop, Industrial, Pasture,
    PermanentCrop, or Residential) in the after map.

    Returns:
        changes: List of dicts with keys: row, col, from_class, to_class
        stats: Dict with summary statistics
    """
    rows, cols = map_before.shape
    changes = []

    total_patches = 0
    forest_before = 0
    forest_after = 0

    for r in range(rows):
        for c in range(cols):
            total_patches += 1
            before_class = map_before[r, c]
            after_class = map_after[r, c]

            if before_class == 'Forest':
                forest_before += 1
            if after_class == 'Forest':
                forest_after += 1

            # Detect deforestation: Forest → non-forest target class
            if before_class == 'Forest' and after_class in DEFORESTATION_TARGETS:
                changes.append({
                    'row': r,
                    'col': c,
                    'from_class': before_class,
                    'to_class': after_class,
                })

    stats = {
        'total_patches': total_patches,
        'forest_before': forest_before,
        'forest_after': forest_after,
        'forest_lost': forest_before - forest_after,
        'deforestation_events': len(changes),
        'deforestation_rate': len(changes) / forest_before if forest_before > 0 else 0,
    }

    return changes, stats


def visualize_land_cover_map(land_cover_map, patch_size, title, output_path):
    """Create a color-coded land cover map visualization."""
    rows, cols = land_cover_map.shape
    img = Image.new('RGB', (cols * patch_size, rows * patch_size))

    for r in range(rows):
        for c in range(cols):
            class_name = land_cover_map[r, c]
            color = CLASS_COLORS.get(class_name, (200, 200, 200))
            # Fill the patch with the class color
            for dy in range(patch_size):
                for dx in range(patch_size):
                    img.putpixel((c * patch_size + dx, r * patch_size + dy), color)

    # Use matplotlib for the final figure with legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(np.array(img))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_patches = [
        mpatches.Patch(color=np.array(color) / 255, label=name)
        for name, color in CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=9, frameon=True, title='Land Cover')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_deforestation(map_before, map_after, changes, patch_size, output_path):
    """
    Create a side-by-side visualization highlighting deforestation areas.
    Left: Before land cover map
    Center: After land cover map
    Right: Change detection (deforestation highlighted in red)
    """
    rows, cols = map_before.shape
    map_width = cols * patch_size
    map_height = rows * patch_size

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Helper to create a color image from a land cover map
    def map_to_image(lcm):
        img = np.zeros((rows * patch_size, cols * patch_size, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                color = CLASS_COLORS.get(lcm[r, c], (200, 200, 200))
                img[r * patch_size:(r + 1) * patch_size,
                    c * patch_size:(c + 1) * patch_size] = color
        return img

    # Before map
    before_img = map_to_image(map_before)
    axes[0].imshow(before_img)
    axes[0].set_title('Before (Time Period 1)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # After map
    after_img = map_to_image(map_after)
    axes[1].imshow(after_img)
    axes[1].set_title('After (Time Period 2)', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Change detection overlay
    change_img = after_img.copy()
    # Dim the background
    change_img = (change_img * 0.4).astype(np.uint8)

    # Highlight deforestation in bright red
    for change in changes:
        r, c = change['row'], change['col']
        change_img[r * patch_size:(r + 1) * patch_size,
                   c * patch_size:(c + 1) * patch_size] = [255, 50, 50]

    # Keep forest areas visible in green
    for r in range(rows):
        for c in range(cols):
            if map_before[r, c] == 'Forest' and map_after[r, c] == 'Forest':
                change_img[r * patch_size:(r + 1) * patch_size,
                           c * patch_size:(c + 1) * patch_size] = [34, 139, 34]

    axes[2].imshow(change_img)
    axes[2].set_title(f'Deforestation Detected ({len(changes)} events)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Add legend to change map
    legend_elements = [
        mpatches.Patch(facecolor=np.array([255, 50, 50]) / 255, label='Deforestation'),
        mpatches.Patch(facecolor=np.array([34, 139, 34]) / 255, label='Remaining Forest'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.suptitle('Satellite-Based Deforestation Detection', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_report(changes, stats, map_before, map_after):
    """Print a detailed deforestation analysis report."""
    print("\n" + "=" * 60)
    print("  DEFORESTATION ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n  Total patches analyzed : {stats['total_patches']}")
    print(f"  Forest patches (before): {stats['forest_before']} ({stats['forest_before'] / stats['total_patches']:.1%})")
    print(f"  Forest patches (after) : {stats['forest_after']} ({stats['forest_after'] / stats['total_patches']:.1%})")
    print(f"  Net forest change      : {stats['forest_lost']:+d} patches")
    print(f"  Deforestation events   : {stats['deforestation_events']}")

    if stats['forest_before'] > 0:
        print(f"  Deforestation rate     : {stats['deforestation_rate']:.1%} of original forest")

    if changes:
        # Breakdown by transition type
        print(f"\n  Transition Breakdown:")
        transition_counts = {}
        for change in changes:
            key = change['to_class']
            transition_counts[key] = transition_counts.get(key, 0) + 1

        for target_class, count in sorted(transition_counts.items(), key=lambda x: -x[1]):
            pct = count / len(changes) * 100
            bar = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
            print(f"    Forest -> {target_class:<20s}: {count:>4d} ({pct:>5.1f}%) [{bar}]")
    else:
        print("\n  No deforestation events detected.")

    # Land cover distribution comparison
    print(f"\n  Land Cover Distribution Comparison:")
    print(f"  {'Class':<24s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
    print(f"  {'-' * 50}")

    for class_name in CLASS_NAMES:
        before_count = np.sum(map_before == class_name)
        after_count = np.sum(map_after == class_name)
        change = after_count - before_count
        change_str = f"{change:+d}" if change != 0 else "--"
        print(f"  {class_name:<24s} {before_count:>8d} {after_count:>8d} {change_str:>8s}")


def save_report_to_file(changes, stats, map_before, map_after, output_path):
    """Save the deforestation report to a text file."""
    with open(output_path, 'w') as f:
        f.write("Satellite-Based Deforestation Detection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total patches analyzed : {stats['total_patches']}\n")
        f.write(f"Forest patches (before): {stats['forest_before']}\n")
        f.write(f"Forest patches (after) : {stats['forest_after']}\n")
        f.write(f"Net forest change      : {stats['forest_lost']:+d}\n")
        f.write(f"Deforestation events   : {stats['deforestation_events']}\n")
        if stats['forest_before'] > 0:
            f.write(f"Deforestation rate     : {stats['deforestation_rate']:.1%}\n")

        if changes:
            f.write(f"\nDeforestation Events:\n")
            for i, change in enumerate(changes):
                f.write(f"  [{i + 1}] Row={change['row']}, Col={change['col']}: "
                        f"{change['from_class']} -> {change['to_class']}\n")

    print(f"  Report saved: {output_path}")


# ----------------------------
# Demo Mode
# ----------------------------

def run_demo(model, patch_size=64):
    """
    Demo mode: simulates the deforestation detection pipeline using
    EuroSAT images stitched into synthetic satellite scenes.

    Creates two synthetic "satellite images":
    - Before: mostly Forest with some other land types
    - After: same scene but with some Forest replaced by AnnualCrop/Residential

    This demonstrates the full pipeline without requiring real Sentinel-2 data.
    """
    print("\n" + "=" * 60)
    print("  DEMO MODE -- Simulating Deforestation Detection")
    print("=" * 60)
    print("  (Using EuroSAT images to build synthetic satellite scenes)\n")

    # Grid size for the synthetic image
    grid_cols, grid_rows = 10, 8

    # Load sample images from EuroSAT
    print("  Loading sample images from EuroSAT dataset...")

    def load_class_images(class_name, count=20):
        """Load sample images from a specific class."""
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  WARNING: {class_dir} not found")
            return []
        all_files = sorted(os.listdir(class_dir))
        selected = all_files[:count]
        images = []
        for fname in selected:
            img_path = os.path.join(class_dir, fname)
            img = Image.open(img_path).convert('RGB').resize((patch_size, patch_size))
            images.append(img)
        return images

    forest_images = load_class_images('Forest', 80)
    crop_images = load_class_images('AnnualCrop', 20)
    residential_images = load_class_images('Residential', 20)
    pasture_images = load_class_images('Pasture', 20)
    river_images = load_class_images('River', 10)
    highway_images = load_class_images('Highway', 10)

    if not forest_images:
        print("ERROR: Cannot find EuroSAT Forest images. Make sure data/EuroSAT/2750/ exists.")
        sys.exit(1)

    random.seed(42)

    # --- Build "Before" scene (mostly forested) ---
    print("  Building 'Before' scene (mostly forested)...")
    before_scene = Image.new('RGB', (grid_cols * patch_size, grid_rows * patch_size))
    before_layout = []

    for r in range(grid_rows):
        row_layout = []
        for c in range(grid_cols):
            # 70% forest, 10% pasture, 5% river, 5% highway, 10% other
            rand = random.random()
            if rand < 0.70:
                img = random.choice(forest_images)
                row_layout.append('Forest')
            elif rand < 0.80:
                img = random.choice(pasture_images) if pasture_images else random.choice(forest_images)
                row_layout.append('Pasture')
            elif rand < 0.85:
                img = random.choice(river_images) if river_images else random.choice(forest_images)
                row_layout.append('River')
            elif rand < 0.90:
                img = random.choice(highway_images) if highway_images else random.choice(forest_images)
                row_layout.append('Highway')
            else:
                img = random.choice(crop_images) if crop_images else random.choice(forest_images)
                row_layout.append('AnnualCrop')

            before_scene.paste(img, (c * patch_size, r * patch_size))

        before_layout.append(row_layout)

    # --- Build "After" scene (some forest cleared) ---
    print("  Building 'After' scene (simulating deforestation)...")
    after_scene = Image.new('RGB', (grid_cols * patch_size, grid_rows * patch_size))
    after_layout = []

    # Create deforestation zones — simulate clearing in specific areas
    deforestation_zones = set()
    # Clear a rectangular area (simulating agricultural expansion)
    for r in range(2, 5):
        for c in range(3, 7):
            deforestation_zones.add((r, c))
    # Clear a few scattered patches (simulating residential development)
    for r, c in [(1, 1), (1, 2), (6, 5), (6, 6), (7, 8)]:
        deforestation_zones.add((r, c))

    for r in range(grid_rows):
        row_layout = []
        for c in range(grid_cols):
            original_class = before_layout[r][c]

            if (r, c) in deforestation_zones and original_class == 'Forest':
                # This forest patch was "cleared"
                if r >= 2 and r <= 4 and c >= 3 and c <= 6:
                    # Agricultural expansion zone
                    img = random.choice(crop_images) if crop_images else random.choice(forest_images)
                    row_layout.append('AnnualCrop')
                else:
                    # Residential development
                    img = random.choice(residential_images) if residential_images else random.choice(forest_images)
                    row_layout.append('Residential')
            else:
                # Unchanged — copy from before
                row_layout.append(original_class)
                x, y = c * patch_size, r * patch_size
                region = before_scene.crop((x, y, x + patch_size, y + patch_size))
                img = region

            after_scene.paste(img, (c * patch_size, r * patch_size))
        after_layout.append(row_layout)

    # Save synthetic scenes
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    before_path = os.path.join(OUTPUT_DIR, "demo_before.png")
    after_path = os.path.join(OUTPUT_DIR, "demo_after.png")
    before_scene.save(before_path)
    after_scene.save(after_path)
    print(f"  Saved synthetic scenes: {before_path}, {after_path}")

    # --- Run the full pipeline ---
    print("\n  Running classification pipeline on 'Before' scene...")
    before_patches, before_grid, _ = split_image_into_patches(before_path, patch_size)
    before_results = classify_patches(model, before_patches)
    map_before, conf_before = build_land_cover_map(before_results, before_grid, patch_size)

    print("\n  Running classification pipeline on 'After' scene...")
    after_patches, after_grid, _ = split_image_into_patches(after_path, patch_size)
    after_results = classify_patches(model, after_patches)
    map_after, conf_after = build_land_cover_map(after_results, after_grid, patch_size)

    # --- Detect changes ---
    print("\n  Detecting deforestation...")
    changes, stats = detect_changes(map_before, map_after)

    # --- Generate visualizations ---
    print("\n  Generating visualizations...")
    visualize_land_cover_map(
        map_before, patch_size,
        'Land Cover Map — Before (Time Period 1)',
        os.path.join(OUTPUT_DIR, "demo_landcover_before.png")
    )
    visualize_land_cover_map(
        map_after, patch_size,
        'Land Cover Map — After (Time Period 2)',
        os.path.join(OUTPUT_DIR, "demo_landcover_after.png")
    )
    visualize_deforestation(
        map_before, map_after, changes, patch_size,
        os.path.join(OUTPUT_DIR, "demo_deforestation_detection.png")
    )

    # --- Generate report ---
    generate_report(changes, stats, map_before, map_after)
    save_report_to_file(
        changes, stats, map_before, map_after,
        os.path.join(OUTPUT_DIR, "demo_deforestation_report.txt")
    )

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE -- All outputs saved to outputs/")
    print("=" * 60)


# ----------------------------
# Real Satellite Image Mode
# ----------------------------

def run_real(model, before_path, after_path, patch_size=64):
    """
    Run deforestation detection on real satellite imagery.

    Args:
        model: Trained classifier
        before_path: Path to the earlier satellite image
        after_path: Path to the later satellite image
        patch_size: Patch size in pixels (default 64)
    """
    print("\n" + "=" * 60)
    print("  DEFORESTATION DETECTION -- Real Satellite Imagery")
    print("=" * 60)

    for path, label in [(before_path, "Before"), (after_path, "After")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} image not found: {path}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Classify before image
    print(f"\n  Processing 'Before' image: {before_path}")
    before_patches, before_grid, before_img = split_image_into_patches(before_path, patch_size)
    before_results = classify_patches(model, before_patches)
    map_before, conf_before = build_land_cover_map(before_results, before_grid, patch_size)

    # Classify after image
    print(f"\n  Processing 'After' image: {after_path}")
    after_patches, after_grid, after_img = split_image_into_patches(after_path, patch_size)
    after_results = classify_patches(model, after_patches)
    map_after, conf_after = build_land_cover_map(after_results, after_grid, patch_size)

    # Validate grid sizes match
    if before_grid != after_grid:
        print(f"WARNING: Grid sizes differ! Before={before_grid}, After={after_grid}")
        print("Using the smaller grid for comparison.")
        min_rows = min(map_before.shape[0], map_after.shape[0])
        min_cols = min(map_before.shape[1], map_after.shape[1])
        map_before = map_before[:min_rows, :min_cols]
        map_after = map_after[:min_rows, :min_cols]

    # Detect changes
    print("\n  Detecting deforestation...")
    changes, stats = detect_changes(map_before, map_after)

    # Visualizations
    print("\n  Generating visualizations...")
    visualize_land_cover_map(
        map_before, patch_size,
        'Land Cover Map -- Before',
        os.path.join(OUTPUT_DIR, "landcover_before.png")
    )
    visualize_land_cover_map(
        map_after, patch_size,
        'Land Cover Map -- After',
        os.path.join(OUTPUT_DIR, "landcover_after.png")
    )
    visualize_deforestation(
        map_before, map_after, changes, patch_size,
        os.path.join(OUTPUT_DIR, "deforestation_detection.png")
    )

    # Report
    generate_report(changes, stats, map_before, map_after)
    save_report_to_file(
        changes, stats, map_before, map_after,
        os.path.join(OUTPUT_DIR, "deforestation_report.txt")
    )

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE -- All outputs saved to outputs/")
    print("=" * 60)


# ----------------------------
# CLI Entry Point
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Detect deforestation from satellite imagery using a trained CNN classifier.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no satellite imagery needed):
  python detect_deforestation.py --demo

  # Real satellite imagery:
  python detect_deforestation.py --before sentinel_2018.tif --after sentinel_2024.tif

  # Custom patch size:
  python detect_deforestation.py --before img1.tif --after img2.tif --patch-size 128
        """
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode using EuroSAT images')
    parser.add_argument('--before', type=str, default=None,
                        help='Path to the earlier satellite image')
    parser.add_argument('--after', type=str, default=None,
                        help='Path to the later satellite image')
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Patch size in pixels (default: 64)')

    args = parser.parse_args()

    if not args.demo and (args.before is None or args.after is None):
        print("ERROR: Please specify either --demo or both --before and --after images.")
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("  SATELLITE DEFORESTATION DETECTION SYSTEM")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    model = load_model()

    if args.demo:
        run_demo(model, patch_size=args.patch_size)
    else:
        run_real(model, args.before, args.after, patch_size=args.patch_size)


if __name__ == "__main__":
    main()
