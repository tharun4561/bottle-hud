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



\## Milestone 3

\*\*Why bottom center?\*\*

Its symmetrical, it's out of the way of the frame, and it's essentially where I am on the frame.



\*\*What would moving the origin change about the target line?\*\*

The line start and length would change if it was put somewhere else on the screen. It wouldn't change the detection, its just a different  reference point.



\*\*How OpenCV image coordinates work:\*\*

0,0 is at the top left corner of the screen and x increases as it goes right and y increases as it goes down.



\*\*Crosshair position on my 640×480 frame:\*\*

x = width//2

bottom margin is 40

y = height - bottom margin

coordinates: (320, 440)





\## Milestone 4

\*\*Why does calibration only need to happen once per camera?\*\*

The camera's focal length doesn't change, so calibrating once is enough.



\*\*What if the bottle is tilted?\*\*

The pixel height will change, so the accuracy will decrease, giving the wrong focal length



\*\*My calibration:\*\*

\- H = 0.142 m, D = 1.00 m, averaged h = \~87.4 px → f = 615.3 px



\## Milestone 5

\*\*Why does the distance jump when the bottle is partly off screen?\*\*

It thinks that the clipped image of the object is just the object, but smaller. So, a smaller object means farther away.



\*\*How could you smooth it?\*\*

Options:

average last N values of h, 1 wrong value can make it inaccurate

Exponential moving avg on h, lags

median of last n values, less smooth



\## Milestone 6

\*\*Why use atan instead of the raw pixel offset?\*\*

The pixel offset is just an image, to turn it into an angle we need to know how far the object is from the lens



\*\*What does the angle read with the bottle centered?\*\*

The angle is 0 degrees horizontal and vertical



\*\*Why the line's slope isn't the bearing:\*\*

the line starts from the crosshair while the bearing is measured from the middle of the camera's optical lens.





\## Milestone 7

\*\*What's slowing the FPS down?\*\*

More power on the laptop increased the fps, and using the GPU made it better. Also, making the IMGSZ = 480 helped



\*\*What could speed it up?\*\*

decreasing the IMGSZ makes the screen smaller and less accurate, but makes the YOLO run faster. I'm already at the fastest YOLO model (nano)





\*\*HUD panel design:\*\*

\- How it stays small and readable: Everything is in the top left.

