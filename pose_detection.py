import cv2
import mediapipe as mp
import time


class pose_detection_module:
	def __init__(self):
		self.image_handler = cv2.VideoCapture(0)
		self.pose_detector = mp.solutions.pose.Pose()
		self.drawer = mp.solutions.drawing_utils
		self.important_landmarks = [11, 12, 13, 14, 23, 24, 27, 28]
		self.counter = 0

	def get_image(self):
		status, frame = self.image_handler.read()

		if status:
			return frame


	def get_landmark_positions(self, frame):
		rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.pose_detector.process(rgb_image)
		landmarks = []

		if results.pose_landmarks:
			for id, landmark in enumerate(results.pose_landmarks.landmark):
				h, w, c = frame.shape
				landmarks.append([int(landmark.x * w), int(landmark.y * h), landmark.z, landmark.visibility])

		return (landmarks, results.pose_landmarks)


	def draw_landmarks(self, frame, landmarks, fps):
		if landmarks:
			self.drawer.draw_landmarks(frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS)

		if fps:
			cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)

		return frame




def main():
	PressUpCounter = pose_detection_module()
	run = True
	current_time, previous_time = 0, 0

	while run:
		current_time = time.time()
		fps = int(1 / (current_time - previous_time))
		previous_time = current_time

		frame = PressUpCounter.get_image()
		positions, results = PressUpCounter.get_landmark_positions(frame)
		processed_frame = PressUpCounter.draw_landmarks(frame, results, fps)

		cv2.imshow('Image', processed_frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			run = False



if __name__ == '__main__':
	main()
