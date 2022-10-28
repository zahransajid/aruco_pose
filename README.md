# Code for aruco detection and pose estimation

Contains code to detect and find the pose for individual markers or markers in a square pattern (included image to print).

To run the code demos,

Linux:

```bash
git clone https://github.com/zahransajid/aruco_pose
cd aruco_pose
source bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows:

```powershell
git clone https://github.com/zahransajid/aruco_pose
cd aruco_pose
Scripts/Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Contains code and some sample input frames
The actual code is in `detect.py` and `camera_pose.py`.

Run main.py to see some demos based on those and the example frames in `frames_limited/`.

Screenshots of the demos running in `screenshots/`.
Also find the printable sheet for the 4 marker tracking page in `tracking_page.png`.
