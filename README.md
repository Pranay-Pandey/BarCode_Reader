# BarCode_Reader
  <p>
    This repository contains a solution for recognizing barcodes on the items in an image containing multiple grocery items with a single-colored background. The solution uses OpenCV and Zbar libraries for barcode recognition and object detection.
  </p>
  <h2>Dependencies</h2>
  <ul>
    <li>OpenCV</li>
    <li>pyzbar</li>
    <li>Numpy</li>
    
<li>dbr</li>
<li>ultralytics</li>
  </ul>
  <h2>Initial Thought</h2>
  <ol>
    <li>Import the required libraries</li>
    <li>Define the function to detect objects in the image using OpenCV's feature-based object detection method</li>
    <li>Define the function to recognize barcodes in the detected objects using Zbar library</li>
    <li>Loop through all the detected objects and use the barcode recognition function to recognize barcodes in each object</li>
    <li>Draw bounding boxes of different colors around the objects based on the results of the barcode recognition: 
      <ul>
        <li>Blue for successfully recognized barcodes</li>
        <li>Yellow for partially recognized barcodes</li>
        <li>Red for objects without barcodes</li>
        <li>Black for each item in the image</li>
      </ul>
    </li>
    <li>Print the value of every barcode and the number of times each barcode appears in the image</li>
  </ol>
  <h2>How to Run</h2>
  <ol>
    <li>Clone or download the repository</li>
    <li>Install the required dependencies</li>
    <li>Run the solution by executing the main file in the terminal or command prompt</li>
  </ol>
  
  <h2> Key Results </h2>
  
  <table>
  <tr>
            <td><img src="https://user-images.githubusercontent.com/79053599/216819562-1b09b62f-9c89-43d4-b397-1b95df9790e1.jpg" width="400" height="400"></td>
            <td><img src="https://user-images.githubusercontent.com/79053599/216819576-3c401c59-24ee-41ba-a58f-5eb8ec3010d1.jpg" width="400" height="400"></td>
         </tr>
         <tr>
            <td><img src="https://user-images.githubusercontent.com/79053599/216819624-8c15020a-1c99-4ce5-bf9d-85e2d166c844.jpg" width="400" height="400"></td>
            <td><img src="https://user-images.githubusercontent.com/79053599/216819640-9fb54cdd-e442-49cf-a6c1-f671095ffb13.jpg" width="400" height="400"></td>
         </tr>
  </table>
<br>
The rest of the results can be found in the Output Folder.
 <h2>Solution Procedure:</h2>
    <p>The following steps are performed in the solution:</p>
    <ul>
      <li>The image is loaded.</li>
      <li>Yolo Object detection is applied on the image and all the objects present in the image are identified.</li>
      <li>The identified objects are cropped and passed to the Fine object getter method which uses clustering of corners present in the image to identify individual items.</li>
      <li>The cropped images of individual items are processed to detect barcode present in them. For barcode detection Pyzbar and Dynamsoft barcode reader libraries are used.</li>
      <li>The images are coloured accordingly to indicate successful barcode detection or not.</li>
      <li>Finally, Barcode detection is applied directly on the whole image to see if there is any left out barcode.</li>
</ul>
<h2>Running the Code</h2>
<h3>From Python</h3>
<p>The main code for barcode identification is present in the file <strong>main.py</strong>. To run the code from python, follow the instructions below:</p>
<ol>
  <li>Make a new python script to get the results</li>
  <pre><code>import main

file = main.main_func("Image_name/path")

img_output = file[0]
barcode_coordinates = file[1]
</code></pre>
</ol>
<h3>From Command Line Interface</h3>
<p>To run the code from the Command Line Interface, follow the instructions below:</p>
<ol>
  <li>Open your terminal or command prompt</li>
  <li>Change the directory to the project directory using the <code>cd</code> command</li>
  <li>Run the following command:<br><code>python main.py [image_file/loc] [optional_arguments]</code></li>
</ol>
<h3>Notebook</h3>
<p>The <strong>basic script.ipynb</strong> file is the raw Jupyter Notebook that was used for training and experimentation and also for finding the hyperparameter's optimal values, etc. You can use this notebook for understanding the implementation of the code and other approches.</p>



  
