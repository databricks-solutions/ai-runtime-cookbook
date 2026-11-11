#!/usr/bin/env python3
"""Generate a large synthetic DICOM benchmark without patient data."""

import os
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def base_images():
    y, x = np.mgrid[0:512, 0:512].astype(np.float32)
    images = []
    for index in range(32):
        offset = (index % 8 - 3) * 5
        left = np.exp(-(((x - 175 + offset) / 90) ** 2 + ((y - 260) / 145) ** 2))
        right = np.exp(-(((x - 337 - offset) / 90) ** 2 + ((y - 260) / 145) ** 2))
        center = np.exp(-(((x - 256) / 42) ** 2 + ((y - 280) / 155) ** 2))
        texture = 180 * np.sin((x + index * 3) / 23) * np.cos((y + index * 5) / 29)
        image = 800 + index * 40 + 8500 * (left + right) + 4500 * center + texture
        images.append(np.clip(image, 0, 65535).astype(np.uint16))
    return images


def main():
    count = int(os.environ["NUM_SYNTHETIC_IMAGES"])
    output_dir = Path(os.environ["INPUT_PATH"])
    output_dir.mkdir(parents=True, exist_ok=False)
    images = base_images()
    study_uid = generate_uid()
    series_uid = generate_uid()

    for index in range(count):
        pixels = np.roll(images[index % len(images)], (index // 32) % 21 - 10, axis=1)
        scale = 0.85 + (index % 17) * 0.02
        pixels = np.clip(pixels.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
        sop_uid = generate_uid()
        output_path = output_dir / f"synthetic_{index:05d}.dcm"

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        dataset = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SpecificCharacterSet = "ISO_IR 100"
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = sop_uid
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.Modality = "DX"
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.SamplesPerPixel = 1
        dataset.Rows = 512
        dataset.Columns = 512
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleIntercept = 0
        dataset.RescaleSlope = 1
        dataset.WindowCenter = 8000
        dataset.WindowWidth = 16000
        dataset.BurnedInAnnotation = "NO"
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(output_path), enforce_file_format=True)

        if (index + 1) % 1000 == 0:
            print(f"Generated {index + 1}/{count} synthetic DICOM files", flush=True)


if __name__ == "__main__":
    main()
