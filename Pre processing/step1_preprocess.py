import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


# Landsat-8 TIRS Band 10 Calibration Constants (Default USGS Values)
RADIANCE_MULT_B10 = 3.3420e-4
RADIANCE_ADD_B10 = 0.1
K1_CONSTANT_B10 = 774.8853
K2_CONSTANT_B10 = 1321.0789

def radiometric_calibration_thermal(thermal_dn, mult=RADIANCE_MULT_B10, add=RADIANCE_ADD_B10, k1=K1_CONSTANT_B10, k2=K2_CONSTANT_B10):
    """
    Convert Raw Landsat 8 TIRS Band 10 Digital Numbers (DN) to
    Top-Of-Atmosphere (TOA) Spectral Radiance and Brightness Temperature (Kelvin).
    """
    dn = np.asarray(thermal_dn, dtype=np.float32)
    # Radiance (W / (m^2 * srm * um))
    radiance = mult * dn + add
    radiance = np.maximum(radiance, 1e-5)  # avoid log(0) or division by zero
    
    # Brightness Temperature in Kelvin
    temperature_k = k2 / np.log((k1 / radiance) + 1.0)
    return temperature_k

def generate_cloud_mask(optical, thermal, qa_band=None):
    """
    Generate cloud mask (1 = Clear land/water, 0 = Cloud/Shadow/Invalid).
    Uses QA band if available, else spectral reflectance & thermal thresholding.
    """
    if qa_band is not None:
        cloud_bit = (qa_band & (1 << 3)) != 0
        shadow_bit = (qa_band & (1 << 4)) != 0
        valid_mask = ~(cloud_bit | shadow_bit)
        return valid_mask.astype(np.uint8)

    opt = np.asarray(optical, dtype=np.float32)
    th = np.asarray(thermal, dtype=np.float32)

    # Determine channel axis for optical
    if opt.ndim == 3:
        if opt.shape[0] == 3:  # (3, H, W)
            brightness = np.mean(opt, axis=0)
        else:  # (H, W, 3)
            brightness = np.mean(opt, axis=-1)
    else:
        brightness = opt

    # Normalize brightness to [0, 1]
    b_max = brightness.max() if brightness.max() > 0 else 1.0
    brightness = brightness / b_max

    # Align spatial dimensions if needed
    if brightness.shape != th.shape:
        factors = (th.shape[0] / brightness.shape[0], th.shape[1] / brightness.shape[1])
        brightness = zoom(brightness, factors, order=1)

    th_min, th_max = th.min(), th.max()
    th_norm = (th - th_min) / ((th_max - th_min) + 1e-8)

    # Cloud condition: high optical brightness & low thermal temperature
    is_cloud = (brightness > 0.70) & (th_norm < 0.30)
    valid_mask = (~is_cloud).astype(np.uint8)
    return valid_mask


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr



def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    import rasterio
    with rasterio.open(tif_path) as src:

        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256x256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64x64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 - preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}x")
    print(f"HR size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}x{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
