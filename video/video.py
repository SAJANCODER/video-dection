import cv2
import easyocr

# Define the text detection language
language = 'en'


def process_video(video_source):
  """
  Processes video from a given source (file path or capture index)

  Args:
      video_source: String representing the video file path or integer for capture index (e.g., 0 for webcam)
  """

  # Initialize the text detector
  reader = easyocr.Reader([language], gpu=False)

  # Detection confidence threshold
  threshold = 0.25

  # Open video capture
  cap = cv2.VideoCapture(video_source)
  if not cap.isOpened():
      print("Error opening video source:", video_source)
      return

  # Only print "Start processing" message once
  print("Start processing video...")

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("End of video or error reading frame")
        break

    # Detect text in the frame
    text_results = reader.readtext(frame)

    # Process detected text (optional, modify as needed)
    detected_texts = []
    for bbox, text, score in text_results:
        if score > threshold:
            # Add check to avoid repeated text printing
            if text not in detected_texts:
                detected_texts.append(text)
                print("Detected Text:", text)  # Print detected text (once per unique text)
            bbox = tuple(map(int, bbox[0])), tuple(map(int, bbox[2]))
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    # Display the frame with bounding boxes and text (without waiting)
    cv2.imshow('Detected Text', frame)
    cv2.waitKey(1)  # Non-zero wait key disables frame-by-frame processing (plays video at normal speed)

  # Release capture and close all windows
  cap.release()
  cv2.destroyAllWindows()

# Example usage with video file
# Replace 'your_video.mp4' with the actual filename of your video
process_video('rit.mp4')

# Example usage with webcam (replace 0 with camera index if using another webcam)
# process_video(0)  # Uncomment this line to use webcam

