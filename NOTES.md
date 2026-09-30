\# Notes: Bottle Interceptor HUD



\## Milestone 0

\*\*Why use a virtual environment instead of installing globally?\*\*

A virtual environment allows isolated sets of packages, so the projects can't break each other. It can also be reproduced on other machines.



\## Milestone 1

\*\*What does cv2.waitKey() do, and why does the window freeze without it?\*\*

It lets the  window respond and update, also it checks if the button is pressed every 1 ms. The waitkey processes all of the messages that are sent from Windows and from imshow, without it, the messages pile up and the window is seen as not responding. 

