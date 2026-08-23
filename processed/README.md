# Processed Data

This directory contains generated outputs from the second-stage processing pipeline.

## Manual Exclusions

After reviewing the results of the face-processing stage, 35 images were manually identified as unsuitable because the processing result was visibly distorted.

The original images were manually removed from the processed output and moved to a separate folder for reference.

To make sure these images are not processed again on another machine, their IDs are stored in:

`ffhq_manually_excluded_after_processing.json`

The processing code was also updated to read this exclusion list and skip these images before running the processing pipeline.

This ensures that if another user runs the code on a machine where the original FFHQ images are still available, the 35 manually excluded images will be skipped automatically.

## Excluded Images

The exclusion list contains 35 unique FFHQ image IDs.

The JSON file is kept separately from `ffhq_failed.json` because these images were not necessarily processing failures detected automatically by the code. They were manually excluded after reviewing the generated results and identifying visible distortion.

## Generated Outputs

The following directories contain generated processing outputs:

- `faces/`
- `landmarks/`
- `masks/`

These generated files are not stored in the GitHub repository because they are produced by the processing scripts.