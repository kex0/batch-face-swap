import platform
import launch

if not launch.is_installed("mediapipe") or not launch.is_installed("mediapipe-silicon"):
    name = "Batch Face Swap"
    if platform.system() == "Darwin":
        # MacOS
        launch.run_pip("install mediapipe-silicon", "requirements for Batch Face Swap for MacOS")
    else:
        launch.run_pip("install mediapipe", "requirements for Batch Face Swap")