import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
import csv


dictionary ={
                    'White':([0, 0, 146], [180, 34, 255]),
                    'Gray':([0, 0, 22], [180, 34, 146]),
                    'Light-red':([0,157, 25], [6,255,255]),
                    'Light-Pink':([0,0, 25], [6,157,255]),
                    'Orange':([6, 33, 168], [23, 255, 255]),
                    'Brown':([6, 33, 25], [23, 255, 168]),
                    'yellow':([23, 33, 25], [32, 255, 255]),
                    'Green':([32, 33, 25], [75, 255, 255]), 
                    'Blue-Green':([75, 33, 25], [90, 255, 255]), 
                    'Blue':([90,33, 45], [123, 255, 255]),
                    'Purple':([123, 112, 25], [155, 255, 255]),
                    'Light-Purple':([123, 33, 25], [155, 125, 255]),                   
                    'Pink':([155,34, 25], [180,225,255]),
                    'Deep-Pink':([175,0, 25], [180,157,255]),
                    'Deep-red':([175,157, 25], [180,255,255]),    
                    'black':([0, 0, 0], [180, 255, 26]),      
                    }  
    

confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416    



parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if(args.device == 'cpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif(args.device == 'gpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / location2[2]
	ppm = 1200
	d_meters = d_pixels / ppm
	print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 24
	speed = d_meters * fps * 3.6
	return speed

def calculate_acceleration(final_velocity, initial_velocity):
    fps = 24
    acceleration = (final_velocity - initial_velocity) / fps
    print("Acceleration:", acceleration, "m/s^2")


def getOutputsNames(net):
   
    layersNames = net.getLayerNames()
    # print(layersNames)
    # print(net.getUnconnectedOutLayers())
    return [layersNames[i-1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, name):

    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    label += (' ' + name)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)

    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1e-2, 0, 0, 0], [0, 1e-2, 0, 0], [0, 0, 5e-2, 0], [0, 0, 0, 5e-2]], np.float32)
cars = []
box_paths = {}
points = []

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        box_id = i  # You can use the loop index as the ID or any other unique identifier

        if box_id not in box_paths:
            box_paths[box_id] = []
       

        name = ''
        cars.append([left,left+width, top, top+height])


        cropped = frame[top:top+height, left:left+width]

        name = getColors(cropped)

        
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, name)

       
        if(i == 0) :
            continue
        
        
        for point in points:
            cv.circle(frame, point, 1, (255, 255, 0), 5)
            
        print("NUMBER OF POINTS : ", len(points))

        x, y, w, h = box
        x_mid = int((x+ x + w) / 2)
        y_mid = int((y+ y + h) / 2)

        x_succ, y_succ, w_succ, h_succ = boxes[i-1]

        x_curr_mid = int((x_succ + x_succ + w_succ) / 2)
        y_curr_mid = int((y_succ + y_succ + h_succ) / 2)

        points.append((x_mid, y_mid))
        box_paths[box_id].append([x_mid,y_mid])

        cv.circle(frame, (x_mid, y_mid), 1, (0, 0, 255), 5)
        cv.line(frame, (x_mid, y_mid), (x_mid, y_mid), (0, 0, 255), 5)
        print(box_paths[box_id])
        
        if(len(box_paths[box_id]) == 1):
            X = box_paths[box_id][-1][0]
            Y = box_paths[box_id][-1][1]
        else:
            X = box_paths[box_id][-2][0]
            Y = box_paths[box_id][-2][1]
        location1 = (X,Y)
        location2 = (x_curr_mid, y_curr_mid)
        speed = estimateSpeed(location1, location2)
        acc = 0
        if(len(box_paths[box_id]) != 1):
            
            prev_speed = box_paths[box_id][-2][2]
            acc = calculate_acceleration(speed, prev_speed)

        box_paths[box_id][-1].append(speed)
        box_paths[box_id][-1].append(acc)
        box_paths[box_id][-1].append(classes[classIds[i]])

        location11 = np.array([[np.float32(x_mid)], [np.float32(y_mid)]])
        kalman.correct(location11)
        prediction = kalman.predict()

        cv.circle(frame, (int(prediction[0]), int(prediction[1])), 1, (0, 0, 255), 5)
        cv.line(frame, (int(prediction[0]), int(prediction[1])), (int(x_curr_mid), int(y_curr_mid)),(255, 255, 0))
        measurement = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
        x_pred, y_pred = prediction[0][0], prediction[1][0]
        
        # Kalman update
        kalman.correct(measurement)
        
        # Draw prediction on frame
        cv.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 255, 0), -1)

        
        cv.putText(frame, str(speed) + " km/hr", (x_mid + 15, y_mid - 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    for box_id, path in box_paths.items():
 
        for i in range(1, len(path)):
            prev_box_info = path[i-1]
            curr_box_info = path[i]

            # prev_center = ((prev_box_info[0] + prev_box_info[2]) // 2, (prev_box_info[1] + prev_box_info[3]) // 2)
            # curr_center = ((curr_box_info[0] + curr_box_info[2]) // 2, (curr_box_info[1] + curr_box_info[3]) // 2)

            # Draw line segment between previous and current centers
            # cv.line(frame, prev_center, curr_center, (0, 0, 255), 2)

    
   
    csv_file = "box_data.csv"

    for key, j in box_paths.items():
        print(key)
        print(j)
    
    print("these were the path")
   
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['BoxID', 'X', 'Y', 'Velocity', 'Acceleration', 'Type'])  

 
        for box_id, path in box_paths.items():
            for step, box_info in enumerate(path):
                writer.writerow([box_id, *box_info])


            writer.writerow([])

        

       
def getColors(image):

    if(image.size == 0) :
        return ''
    
    
    image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    dictionary ={
                    'White':([0, 0, 116], [57, 57, 255]),
                    'Red':([180,0,0], [255,255,255]),
                    'Light-red':([0,38, 56], [10,255,255]),
                    'orange':([10, 38, 71], [20, 255, 255]),
                    'yellow':([18, 28, 20], [33, 255, 255]),
                    'green':([36, 10, 33], [88, 255, 255]), 
                    'blue':([87,32, 17], [120, 255, 255]),
                    'purple':([138, 66, 39], [155, 255, 255]),
                    'Deep-red':([170,112, 45], [180,255,255]),
                    'Deep-blue': ([110,100,100], [120,255,255]),
                    'black':([0, 0, 0], [179, 255, 50]),      
                    }  
    
    color_name = []
    color_count =[]
             

    for key,(lower,upper) in dictionary.items():
        
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
     
        mask = cv.inRange(image_HSV, lower, upper)
        
        count = cv.countNonZero(mask)
        
        color_count.append(count)
        
        color_name.append(key)
    
   
    color_count_array = np.array(color_count)
    
    idx = np.argmax(color_count_array)

    color = color_name[idx]

    
    return color



winName = 'VISSIM data extractor'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
   
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
  
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:

    cap = cv.VideoCapture(0)


if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    

    hasFrame, frame = cap.read()
    
   
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
  
        cap.release()
        break


    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)


    outs = net.forward(getOutputsNames(net))

 
    postprocess(frame, outs)


    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
   

    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

