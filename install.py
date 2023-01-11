import launch

if not launch.is_installed("mediapipe"):
    launch.run_pip("install mediapipe==0.9.0.1", "requirements for Batch Face Swap")