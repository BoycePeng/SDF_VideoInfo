import cv2
cap = cv2.VideoCapture('data/20221013091008cut_515_820.mp4')
print(f'Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
print(f'Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}')
print(f'Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
cap.release()
