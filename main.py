# Imports
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from PIL import Image
import pyzbar.pyzbar as pyzbar
from dbr import *
from threading import Thread
from ultralytics import YOLO
import argparse

# Show Image function
def show_image(img):
    total_image = Image.fromarray(img)
    total_image.show()

# Object Detection Model
model = YOLO("yolov8n.pt")

def Get_Broad_Objects(file, conf=0.01, iou=0.4, cutoff_factor=0.6, cutoff_limit=0.0004,
                      print_broad_objects_output=False, show_broad_objects_output=False):
    '''
    This function extracts the bounding boxes from the image using YOLO object detection.

    Parameters:
        file (str/ndarray): Name of the image or OpenCV image file.
        conf (float): Confidence threshold for object detection. The default value is experimentally derived.
        iou (float): Intersection over union threshold value.
        cutoff_factor (float): Largest possible area ratio of bounding box to whole image.
        cutoff_limit (float): Limiting smallest possible area ratio of bounding box to whole image.
        print_output (bool): Whether to print text output or not.
        show_output (bool): Whether to show the image output with bounding boxes or not.

    Returns:
        list: A list of bounding boxes coordinates representing a particular object in the image.
    '''
    total_images_bounding_box = []

    if type(file) == str:
        img = cv2.imread(file)
    else:
        img = file
    width, height = img.shape[:2]
    ori_image = img.copy()
    results = model(file, conf=conf, iou=iou)  # predict (Object Detection) on an image

    # NMS
    boxes = []
    confidences = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.detach().numpy()[0]
            boxes.append([x1, y1, x2, y2])
            confidences.append(box.conf.detach().numpy()[0])

    # convert boxes and confidences to numpy array
    boxes = np.array(boxes)
    confidences = np.array(confidences)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, iou)
    indices = indices.flatten()         # selected objects

    selected_boxes = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        area_ratio = abs(x1-x2)*abs(y1-y2)/(width*height)

        # Check if the current bounding box is fully contained in any of the selected boxes
        contained = False
        for selected_box in selected_boxes:
            if x1 >= selected_box[0] and y1 >= selected_box[1] and x2 <= selected_box[2] and y2 <= selected_box[3]:
                contained = True  
                break

        # Checking area_ratio should be in permissible range
        if (area_ratio < cutoff_factor and area_ratio > cutoff_limit) and not contained:
            if print_broad_objects_output:
                print("area ratio = ", area_ratio)
                print(confidences[i])
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            total_images_bounding_box.append([int(x1), int(y1), int(x2), int(y2)])
            selected_boxes.append([x1, y1, x2, y2])
        else:       # Outside permissible range
            # img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            # img = cv2.putText(img, str(confidences[i]),( (int(x1), int(y1))), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color = (0,0,255))
            if print_broad_objects_output:
                print("left ",confidences[i])       

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (show_broad_objects_output):
        t = Thread(target=show_image, args=(img,))
        t.start()
    return total_images_bounding_box



def Get_Fine_Object(file, point, eps=140, min_samples=40, show_fineObjects_output = False):
    '''
    This function detects and returns the fine objects (as in after Broad Objects function) 
    from an image using DBSCAN clustering of the corners detected in the image. These clusters then
    should represent an individual object (grocery item here).

    parameters:
    file - the name of the image or an opencv image file
    point - the bounding box coordinates of the area to be processed (returned from Braod Objects)
    eps - the maximum distance between two samples for one to be considered as in the neighborhood of the other
    (default is 140)
    min_samples - the number of samples in a neighborhood for a point to be considered as a core point (default is 40)
    show_output - flag to show the output image with bounding boxes

    return type:
    output_points - a list of bounding boxes coordinates representing the fine objects in the image in global coordinates.
    '''
    if (type(file)==str):
        img = cv2.imread(file)
    else:
        img = file
    x1, y1, x2, y2 = point
    img = img[y1:y2, x1:x2]             # The area to be processed 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)        # Find corners in the image
    width,height = img.shape[:2]

    output_points = []

    # Get coordinates of corners
    coords = np.column_stack(np.where(dst > 0.01 * dst.max()))
    img_temp = img.copy()
    img_temp[dst>0.01*dst.max()]=[255,255,0]

    # Perform DBSCAN clustering to differntiate different objects
    db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(coords)

    # Get labels for each corner
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Draw bounding box around each cluster
    for i in range(n_clusters_):
        
        # Get indexes of corners in current cluster
        indexes = np.where(labels == i)

        # Get coordinates of corners in current cluster
        cluster_coords = coords[indexes]

        # Get min and max x and y coordinates
        min_x, min_y = np.min(cluster_coords, axis=0)
        max_x, max_y = np.max(cluster_coords, axis=0)
        if abs((max_x-min_x)*(max_y-min_y))/(width*height)<0.1:     # Object is too small
            continue
        output_points.append([min_y+x1, min_x+y1, max_y+x1, max_x+y1])
        # Draw bounding box
        # cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)
        # print("area ratio = ",abs((max_x-min_x)*(max_y-min_y))/(width*height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (show_fineObjects_output):
        t = Thread(target=show_image, args=(img,))
        t.start()

    return output_points

def calculate_iou(bbox1, bbox2):
    '''
    This function calculates the Intersection Over Union (IoU) between two bounding boxes, bbox1 and bbox2.

    Parameters:
    bbox1 (list): a list of four integers representing the top-left (x1, y1) and bottom-right 
    (x2, y2) coordinates of the first bounding box.
    bbox2 (list): a list of four integers representing the top-left (x3, y3) and bottom-right 
    (x4, y4) coordinates of the second bounding box.

    Returns:
    iou (float): the Intersection Over Union value, calculated as the ratio of the intersection
    area of the two bounding boxes to the union area of the two bounding boxes. 
    If there is no overlap, returns 0.0.
    '''
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # find the coordinates of the intersection rectangle
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    # check if there is an overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)

    # calculate the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def getNdraw_barcode(image, iou_threshold=0.5, show_barcode_output=False, only_barcode=False):
    """
    Detect and extract barcode information from the given image.

    Args:
        image (numpy.ndarray): Input image.
        iou_threshold (float, optional): Threshold for IoU overlap between two bounding boxes. Defaults to 0.5.
        show_output (bool, optional): Flag to indicate whether to display the image with detected barcodes. Defaults to False.
        save_image_output (bool, optional): Flag to indicate whether to save the image with detected barcodes. Defaults to False.
        only_barcode (bool, optional): Flag to indicate whether to only detect and extract EAN13 barcodes. Defaults to False.

    Returns:
        list: List of barcode information, including the bounding box coordinates (x, y, x2, y2) and barcode text.
    """
    # Create a copy of the original image
    img_ori = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize a list to store the barcode information
    information = []

    # Detect barcodes using Dynamsoft Barcode Reader
    BarcodeReader.init_license("license_key")
    reader = BarcodeReader()
    Image.fromarray(img_ori).save('temporary_image.png')
    try:
        # Decode barcodes in the image
        text_results = reader.decode_file('temporary_image.png')

        # If barcodes are detected
        if text_results != None:
            # Loop over the barcode results
            for text_result in text_results:
                # Get the barcode text and localization points
                text = text_result.barcode_text
                points = text_result.localization_result.localization_points

                # Skip non-EAN13 barcodes if only_barcode flag is set
                if (text_result.barcode_format_string != 'EAN_13' and only_barcode):
                    continue

                # Convert the localization points to a numpy array
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))

                # Get the bounding box coordinates (x, y, x2, y2)
                x, y, w, h = cv2.boundingRect(pts)
                x2, y2 = x + w, y + h

                # Check if the barcode information is already in the list
                if [x, y, x2, y2, text] not in information:
                    # Add the barcode information to the list
                    information.append([x, y, x2, y2, text])
                    
                    # Draw the bounding box on the image
                    cv2.polylines(image, [pts], True, (255, 0, 0), 2)
                    
    except BarcodeReaderError as bre:
        # Print error message
        print(bre)

    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
        barcodeData = barcode.data.decode("utf-8")
        (x, y, w, h) = barcode.rect
        boxA = [x, y, x+w, y+h]
        if (barcode.type!="EAN13" and only_barcode):
            # Continue to next iteration if the barcode type is not EAN13 and only_barcode is True
            continue

        # Check if the barcode information is already in the list
        if [x, y, x+w, y+h, barcodeData] not in information:
            # Calculate IoU for the current bounding box from pyzbar and existing bounding boxes from Dynamsoft
            overlap = False
            for info in information:
                x1, y1, x2_, y2_, _ = info
                boxB = [x1, y1, x2_, y2_]
                iou = calculate_iou(boxA, boxB)
                if iou >= iou_threshold:
                    overlap = True
                    # Update the barcode data if there is overlap
                    info[4]=barcodeData
                    break
            if not overlap:
                # Add the barcode information to the list if there is no overlap
                information.append([x, y, x+w, y+h, barcodeData])
                # Draw a rectangle around the barcode
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
    # Convert the image back to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Show the image if show_output is True
    if show_barcode_output:
        t = Thread(target=show_image, args=(image,))
        t.start()

    os.remove('temporary_image.png')    # remove the temporary image
    # Return the barcode information
    return information

def is_overlap(box1, box2):
    """
    Determine if two boxes overlap with each other.
    
    Parameters:
    box1 (tuple): The coordinates of the first box represented as (x1, y1, x2, y2)
    box2 (tuple): The coordinates of the second box represented as (x1g, y1g, x2g, y2g)

    Returns:
    bool: True if boxes overlap, False otherwise.
    """
    # Unpack the coordinates of the two boxes
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate the minimum and maximum x and y coordinates for each box
    x1 = min(x1, x2)
    y1 = min(y1, y2)
    x2 = max(x1, x2)
    y2 = max(y1, y2)
    
    x1g = min(x1g, x2g)
    y1g = min(y1g, y2g)
    x2g = max(x1g, x2g)
    y2g = max(y1g, y2g)

    # Check if the two boxes overlap vertically
    if x1 >= x2g or x2 <= x1g:
        return False
    
    # Check if the two boxes overlap horizontally
    if y1 >= y2g or y2 <= y1g:
        return False
    
    # If the two boxes overlap vertically and horizontally, return True
    return True



# MAIN FUNCTION
def main_func(image_file_name, show_out = False, use_outer = False, check_directly=False , 
broad_conf = 0.01, broad_iou = 0.4, broad_cutoff = 0.6, broad_limit = 0.0004, broad_print = False, broad_show = False,
fine_eps = 140, fine_min_samples = 40 , show_fineObjects_output = False,
barcode_iou = 0.1, show_barcode_output = False, only_barcode = False ):
    """
    Main function that initiates the entire process of finding and classifying barcodes. 
    It returns an array containing the processed image and the global barcode coordinates 
    and the barcode information.

    Parameters:
    img_file (str): Path to the image file to be processed
    use_outer (bool, optional): Boolean flag to use the inner objects detected or consider the outer boxes 
    from object detection to be the objects.
    show_out (bool, optional): Boolean flag to indicate if the processed image should be displayed. Default is False.
    check_directly (bool, optional): Boolean flag to direclty check for barcodes from the image without detecting objects in the image. 
    Default is False.
    Other parameters are defined in the individual functions

    The barcode information in printed

    Returns:
    np.array: An array containing the processed image and the global barcode coordinates.

    """
    if not os.path.exists(image_file_name):
        print("Image file does not exist at ", image_file_name)
        exit(0)
    # Read the image file
    image = cv2.imread(image_file_name)

    # Get outer boxes (broad objects) in the image
    list_of_outer_boxes = Get_Broad_Objects(image_file_name, conf=broad_conf, iou=broad_iou, 
                    cutoff_factor=broad_cutoff,  cutoff_limit=broad_limit,
                    print_broad_objects_output=broad_print, show_broad_objects_output=broad_show )

    # Get inner boxes (fine objects) in the image
    list_of_inner_boxes = []
    for box in list_of_outer_boxes:
        list_of_inner_boxes.append(Get_Fine_Object(image_file_name,box ,  eps=fine_eps, 
        min_samples=int(fine_min_samples), show_fineObjects_output = show_fineObjects_output))

    # Initialize the list to store the coordinates of objects
    cordinates_of_objects = []

    # Draw black bounding boxes around either the outer or inner objects
    if (use_outer):
        for index in range(len(list_of_outer_boxes)):
            x1, y1 , x2, y2 = list_of_outer_boxes[index]
            image = cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,0), 2)
            cordinates_of_objects.append([x1, y1 , x2, y2])
    else:
        for index in range(len(list_of_inner_boxes)):
            for k in list_of_inner_boxes[index]:
                x1, y1 , x2, y2 = k
                image = cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,0), 2)
                cordinates_of_objects.append([x1, y1 , x2, y2])

    # Get barcode data from individual object
    info = []
    for box in cordinates_of_objects:
        img_to_send = image[box[1]:box[3], box[0]:box[2]]
        info.append(getNdraw_barcode(img_to_send, iou_threshold=barcode_iou, 
        show_barcode_output=show_barcode_output, only_barcode=only_barcode))

    # Get the global coordinates of barcode
    global_bar_codecordinates = []

    # Loop through the length of the barcode information obtained
    for i in range(len(info)):
        # If there is no barcode information in the current iteration, append an empty list to the global barcode coordinates list
        if (len(info[i]) == 0):
            global_bar_codecordinates.append([])
            # Draw a red rectangle around the object in the image using the object's coordinates
            image = cv2.rectangle(image, (int(cordinates_of_objects[i][0]),int(cordinates_of_objects[i][1])), 
            (int(cordinates_of_objects[i][2]),int(cordinates_of_objects[i][3])), (0,0,255), 2)
            # Continue to the next iteration
            continue
        # Otherwise, calculate the global barcode coordinates
        for coord in info[i]:
            x1, y1, x2, y2 = cordinates_of_objects[i][0] + coord[0], cordinates_of_objects[i][1] + coord[1], \
            cordinates_of_objects[i][0] + coord[2], cordinates_of_objects[i][1] + coord[3]
            global_bar_codecordinates.append([x1, y1, x2, y2])

    # Create a dictionary to store the unique barcodes
    Objects = {}

    # Loop through the barcode information
    for bar_code_info in info:
        for elements in bar_code_info:
            # If the barcode already exists in the dictionary, increment its count
            if (elements[4] in Objects):
                Objects[elements[4]] += 1
            else:
                # Otherwise, add the barcode to the dictionary with a count of 1
                Objects[elements[4]] = 1

    # Check if any barcodes were left out and need to be directly found from the image
    if (check_directly):
        # Read the image file
        img_current = cv2.imread(image_file_name)
        # Get the barcode information from the image
        outer_info = getNdraw_barcode(img_current, iou_threshold=barcode_iou, 
        show_barcode_output=show_barcode_output, only_barcode=only_barcode)
        # Loop through the barcode information
        for daa in outer_info:
            x1, y1, x2, y2 = daa[:4]
            overlap = False
            # Loop through the global barcode coordinates
            for g_barcode in global_bar_codecordinates:
                # If the current global barcode coordinates have length 4
                if (len(g_barcode) == 4):
                    x1g, y1g, x2g, y2g = g_barcode
                    # Check if there is overlap between the current barcode and the global barcode
                    if is_overlap([x1, y1, x2, y2], [x1g, y1g, x2g, y2g]):
                        overlap = True
                        break
            # If there is no overlap, draw a blue rectangle around the barcode and add it to the dictionary of barcodes
            if not overlap:
                image = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0),2 )
                global_bar_codecordinates.append([x1, y1, x2, y2])
                Objects[daa[4]]=1
                if (daa[4] in Objects):
                    Objects[daa[4]]+=1
                else:
                    Objects[daa[4]]=1
                
    # Showing the final image with all the barcodes 
    if (show_out):
        # Start a new thread to show the image
        t = Thread(target=show_image, args=(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),))
        t.start()

    # Printing the unique barcodes present in the image
    print(Objects)

    # Return the final image and global barcode coordinates
    return np.array([image, global_bar_codecordinates],dtype=object)

# Example Usage
# main_func(r'Input_Images\all barcode\IMG_20220303_173611.jpg', show_out=True)
# main_func(r'Input_Images\all barcode\IMG_20220303_173846.jpg', True, check_directly=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_name", type=str, help="Name of the image file")
    parser.add_argument("--show_out", type=bool, default=True, help="Display image eith the final output")
    parser.add_argument("--use_outer", type=bool, default=False, help="Use outer bounding boxes instead of inner")
    parser.add_argument("--check_directly", type=bool, default=False, help="Use direct detection method")
    parser.add_argument("--broad_conf", type=float, default=0.01, help="Confidence threshold for broad detection")
    parser.add_argument("--broad_iou", type=float, default=0.4, help="IoU threshold for broad detection")
    parser.add_argument("--broad_cutoff", type=float, default=0.6, help="Cutoff threshold for broad detection")
    parser.add_argument("--broad_limit", type=float, default=0.0004, help="Limit for broad detection")
    parser.add_argument("--broad_print", type=bool, default=False, help="Print broad detection results")
    parser.add_argument("--broad_show", type=bool, default=False, help="Show broad detection results")
    parser.add_argument("--fine_eps", type=int, default=140, help="Epsilon value for fine detection")
    parser.add_argument("--fine_min_samples", type=int, default=40, help="Min samples for fine detection")
    parser.add_argument("--show_fineObjects_output", type=bool, default=False, help="Show fine objects detection results")
    parser.add_argument("--barcode_iou", type=float, default=0.1, help="IoU threshold for barcode detection")
    parser.add_argument("--show_barcode_output", type=bool, default=False, help="Show barcode detection results")
    parser.add_argument("--only_barcode", type=bool, default=False, help="Only detect barcodes")
    args = parser.parse_args()
    main_func(args.image_file_name, args.show_out, args.use_outer, args.check_directly, 
                args.broad_conf, args.broad_iou, args.broad_cutoff, args.broad_limit, args.broad_print, args.broad_show, args.fine_eps,
                args.fine_min_samples, args.show_fineObjects_output, args.barcode_iou, args.show_barcode_output, args.only_barcode)


