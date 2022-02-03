import online_video_processing

options = {
    'model': "cfg/tiny-yolo-voc-1c.cfg",
    'load': 1000,
    'threshold': 0.2
}

online_video_processing.process_stream(options, 0)

'''
Below are the instructions for an image processing for potholes and 
drawing boxes around it.
'''

tfNet = TFNet(options)

img = cv2.imread('demo2.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfNet.return_predict(img)

top_left = [(result[i]['topleft']['x'], result[i]['topleft']['y'])
            for i in range(len(result))]
bottom_right = [(result[i]['bottomright']['x'], result[i]
                 ['bottomright']['y']) for i in range(len(result))]
labels = [result[i]['label'] for i in range(len(result))]
for i in range(len(top_left)):
    img = cv2.rectangle(img, top_left[i], bottom_right[i], (0, 255, 0), 7)
    img = cv2.putText(img, labels[i], top_left[i],
                      cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 4)
plt.imshow(img)
plt.show()
