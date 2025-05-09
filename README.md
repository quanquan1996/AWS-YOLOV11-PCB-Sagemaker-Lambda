# PCB Defect Detection

This repository contains a PCB defect detection application using YOLOv11 model. The application can detect various defects in PCB images and videos.

## Recent Changes

The following changes were made to fix the issue where the inference was not detecting defects:

1. **Added Confidence Threshold to Image Processing**:
   - Modified the `process_image()` function to use a lower confidence threshold (0.3) to increase detection sensitivity
   - This makes the image processing more consistent with video processing and helps detect more defects

2. **Removed Duplicate Model Loading**:
   - Removed the redundant model loading in the `process_video()` function to use the already loaded model
   - This improves efficiency and consistency

3. **Fixed Model Path Inconsistency**:
   - Updated the Dockerfile to copy the correct model file name (`bestv11.pt` instead of `best.pt`)
   - This ensures the model is correctly referenced and loaded without errors

4. **Updated Test Script**:
   - Modified test.py to use relative paths or environment variables for test images and videos
   - This improves portability and makes testing easier
   - Fixed the event structure to match what lambda_function.py expects

5. **Added Unit Tests**:
   - Created test_unit.py with unit tests to verify the changes
   - Tests include checking if the correct confidence threshold is used and if duplicate model loading is avoided

## Defect Types

The model can detect the following types of PCB defects:
- Missing hole
- Mouse bite
- Open circuit
- Short
- Spur
- Spurious copper

## Usage

### Testing

To test the application:

1. Place a PCB image in the current directory and name it "test_image.jpg"
2. Or set the TEST_IMAGE_PATH environment variable to point to your test image
3. Run the test.py script:

```bash
export TEST_IMAGE_PATH=/path/to/your/pcb_image.jpg
python test.py
```

### Running Unit Tests

To run the unit tests:

```bash
python test_unit.py
```

### Building Docker Image

To build the Docker image:

```bash
docker build -t pcb-defect-detection .
```

## API

The application provides an API with the following endpoints:

- **POST /pcb**: Process an image or video for PCB defect detection
  - Request body:
    - `file_type`: "image" or "video"
    - `file_data`: Base64-encoded image or video data
  - Response:
    - `processed_image`: Base64-encoded image with defects highlighted (for images)
    - `sample_frames`: Array of Base64-encoded frames with defects highlighted (for videos)
    - `defects`: Array of detected defects with type and confidence
    - `defect_count`: Number of defects detected
    - `defect_summary`: Summary of defect types and counts