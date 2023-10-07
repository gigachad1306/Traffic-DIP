# Traffic-DIP
This tool provides the future trajectory of vehicles and shows their trajectories along with characteristics such as position, velocity, and acceleration. The colour detection is done using KNN. The object detection is based on YOLOv3, and the trajectory prediction is based on Kalman Filter.
<br>
<h1>Technologies Used</h1>
<ul>
  <li>OpenCV</li>
  <li>Yolo V3</li>
  <li>Python</li>
  <li>Kalman Filter </li>
</ul>
<br>
<h1>Usage</h1>
The tool is extremely simple to use. The user has to update the video or image.
<p>
 <ul>
   <li>The user can choose from video and image input. The commands are given as 
     
     python3 main.py --video path 
     python3 main.py --image path
     
   </li>
    <li>
      The output is stored as filename.output.extension, and the CSV dump is found as data.csv
   </li>
   
   <li>
The CSV file has the data as the x-coordinate, y-coordinate, speed, acceleration, and object type of the vehicle. Each vehicle has a unique ID, and new lines in the CSV separate the data for vehicles.
   </li>
  </ul>
  </p>
<br>
<h1>Tool Snapshots</h1>
A sample shot of a CSV file

![image](https://github.com/ShisuiMadara/VISCAL-data/assets/77777434/049c0fa4-aab7-4ac4-9d69-ccd097201867)

extraction and analysis in action. The series of blue points is the path traced by the object. The green lines are the trajectory estimated using the Kalmaan filter. The box has colour detected using KNN and the vehicle type using Yolo V3.
![image](https://github.com/ShisuiMadara/VISCAL-data/assets/77777434/5bb5e9a9-f997-4fe9-96df-51c07f651106)
