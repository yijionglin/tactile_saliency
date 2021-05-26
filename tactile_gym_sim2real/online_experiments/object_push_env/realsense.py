import os
import numpy as np
import cv2
import cv2.aruco as aruco
import time

from robopush.camera import RSCamera, ColorFrameError, DepthFrameError, DistanceError
from robopush.detector import ArUcoDetector
from robopush.tracker import ArUcoTracker, display_fn, NoMarkersDetected, MultipleMarkersDetected

RESOLUTION = (640, 480)

def make_realsense():
    return RSCamera(color_size=RESOLUTION, color_fps=60, depth_size=RESOLUTION, depth_fps=60)

def simple_camera_example():
    try:
        with make_realsense() as camera:
            while True:
                t=time.time()
                camera.read()
                print(time.time()-t)
                cv2.imshow("Color image", camera.color_image)
                cv2.imshow("Depth image", camera.colorized_depth_image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()

def simple_tracker_example():
    try:
        with make_realsense() as camera:
            detector = ArUcoDetector(camera, marker_length=25.0, dict_id=cv2.aruco.DICT_7X7_50)
            tracker = ArUcoTracker(detector, track_attempts=30, display_fn=display_fn)
            while True:
                t = time.time()
                try:
                    tracker.track()
                except (ColorFrameError, DepthFrameError, DistanceError, \
                        NoMarkersDetected, MultipleMarkersDetected) as e:
                    cv2.imshow("Tracking error image", aruco.drawDetectedMarkers(camera.color_image,
                                                                                 detector.rejected,
                                                                                 borderColor=(100, 0, 240)))
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                with np.printoptions(precision=2, suppress=True):
                    print(f"detector.corners: {tracker.corners}, detector.ids: {tracker.ids}")
                    print(f"detector.rvecs: {tracker.rvec}, detector.tvecs: {tracker.tvec}")
                    print(f"detector.poses: {tracker.pose}")
                    print(f"detector.centroids: {tracker.centroid}")
                    print(f"detector.centroid_positions: {tracker.centroid_position}")

                print(time.time()-t)

            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass

    finally:
        cv2.destroyAllWindows()

def track_object(fps=10.0):

    # make save dir
    rs_save_dir = os.path.join(
        'collected_data',
        'temp_rs_data'
    )
    os.makedirs(rs_save_dir, exist_ok=True)

    # save data
    rs_video_file = os.path.join(rs_save_dir, 'rs_video.mp4')

    # setup video writer
    # rs_vid_out = cv2.VideoWriter(
    #     rs_video_file,
    #     # cv2.VideoWriter_fourcc('M','J','P','G'),
    #     cv2.VideoWriter_fourcc(*'MP4V'),
    #     fps,
    #     RESOLUTION
    # )

    # setup the realsense camera for capturing qunatitative data
    try:
        with make_realsense() as rs_camera:
            rs_detector = ArUcoDetector(rs_camera, marker_length=25.0, dict_id=cv2.aruco.DICT_7X7_50)
            rs_tracker = ArUcoTracker(rs_detector, track_attempts=30, display_fn=display_fn)

            rs_rgb_frames = []
            obj_centroids = []

            # main capture loop
            control_rate = 1./fps
            fps_start_time = time.perf_counter()
            while True:
                fps_next_time = fps_start_time + control_rate

                # do stuff
                try:
                    rs_tracker.track()
                except (ColorFrameError, DepthFrameError, DistanceError, \
                        NoMarkersDetected, MultipleMarkersDetected) as e:
                        break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                rs_rgb_frame = rs_tracker.detector.camera.color_image
                obj_centroid = rs_tracker.centroid_position
                rs_rgb_frames.append(rs_rgb_frame.copy()) # copy is important
                obj_centroids.append(obj_centroid)
                print('Object Centroid: ', obj_centroid)

                # write frame to output vid
                # rs_vid_out.write(rs_rgb_frame)

                # hold main loop until next cycle ready
                while time.perf_counter() < fps_next_time:
                    pass

                print("FPS: ", 1.0 / (time.perf_counter() - fps_start_time))
                fps_start_time = fps_next_time

            # finishing
            # rs_vid_out.release()


    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # simple_camera_example()
    # simple_tracker_example()
    track_object()
