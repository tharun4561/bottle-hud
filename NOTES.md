\# Notes: Bottle Interceptor HUD



\## Milestone 0

\*\*Why use a virtual environment instead of installing globally?\*\*

A virtual environment allows isolated sets of packages, so the projects can't break each other. It can also be reproduced on other machines.



\## Milestone 1

\*\*What does cv2.waitKey() do, and why does the window freeze without it?\*\*

It lets the  window respond and update, also it checks if the button is pressed every 1 ms. The waitkey processes all of the messages that are sent from Windows and from imshow, without it, the messages pile up and the window is seen as not responding.



\## Milestone 2

\*\*What does YOLO return for each detection?\*\*

classID, confidence, box coordinates



\*\*What changes when the confidence threshold goes to 0.2 or 0.8?\*\*

From 0.2, most bottles get detected. But, when the threshold is increased to 0.5 or 0.8, the detection flickers a lot and the bottle needs to be in the correct orientation, or position relative to the camera to be detected. 



\*\*Measured confidence on my bottle:\*\*

\- Typical range: \[0.2–0.68]

\- Bottle type: \[tin, stainless steel]

\- Conditions: \[white background, lots of light]



\*\*Observations:\*\*

\- At 0.2: \[Decently steady box, orientation didn't matter, little flickering]

\- At 0.5: \[steady box close to camera, flickers regularly]

\- At 0.8: \[box rarely showed up, flickered at times, worked best when close to screen]



\*\*Threshold I chose and why:\*\*

\[0.3, bottles were rated at 0.3 confidence most times. Filters out bottle shaped objects, but still kept in actual bottles]



\*\*Speed:\*\* about 88 ms per frame (\~11 FPS) with yolo26n on CPU

