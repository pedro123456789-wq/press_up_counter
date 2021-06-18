import cv2
import numpy as np
from pose_detection import pose_detection_module
from math import sqrt, pi, acos



class exercise_counter:
	def __init__(self):
		self.image_handler = cv2.VideoCapture(0)
		self.image_handler.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.image_handler.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
		
		self.pose_detector = pose_detection_module()
		
		self.max_angle = 140
		self.min_angle = 55
		
		self.repetitions = 0
		self.has_dropped = False

		self.up_color = (247, 210, 0)
		self.down_color = (45, 45, 237)
		self.color = self.down_color


	def take_picture(self):
		status, frame = self.image_handler.read()

		if status:
			return True, frame
		else:
			return False, '' 



	def calculate_angle(self, x1, y1, x2, y2, x3, y3):
		try:
			vector1 = [x2 - x1, y2 - y1]
			vector2 = [x3 - x2, y3 - y2]

			cos_theata = np.dot(vector1, vector2) / (sqrt(np.dot(vector1, vector1)) * sqrt(np.dot(vector2, vector2)))

			return 180 - int((acos(cos_theata) / pi) * 180)

		except Exception as e:
			print(e)
			return None


	def draw_joint_circles(self, frame, landmarks):
		if len(landmarks) > 0:
			for id in [12, 14, 16, 24, 26, 28]:
				cv2.circle(frame, (landmarks[id][0], landmarks[id][1]), 10, (0, 0, 255), -1)
				cv2.circle(frame, (landmarks[id - 1][0], landmarks[id - 1][1]), 10, (0, 0, 255), -1)

		return frame



	def is_press_up_position(self, landmarks, frame):
		if len(landmarks) > 0:
			frame_height, frame_width, channels = frame.shape

			hip_x = (landmarks[24][0] + landmarks[23][0]) / 2
			shoulder_x = (landmarks[12][0] + landmarks[11][0]) / 2
			ankle_x = (landmarks[28][0] + landmarks[27][0]) / 2

			if ankle_x > hip_x and hip_x > shoulder_x:
				hip_y = (landmarks[24][1] + landmarks[23][1]) / 2
				shoulder_y = (landmarks[12][1] + landmarks[11][1]) / 2
				knee_y = (landmarks[25][1] + landmarks[26][1]) / 2
				toe_y = (landmarks[31][1] + landmarks[32][1]) / 2
				
				if (abs(hip_y - shoulder_y) < 200):
					#add minimum difference between toe and hip
					if toe_y - knee_y > 0:
						return True 

		return False


	def draw_progress_bar(self, frame, angle, color):
		angle_percentage = (angle - self.min_angle) / (self.max_angle - self.min_angle)

		cv2.rectangle(frame, (50, 150), (85, 400), color, 3)
		cv2.rectangle(frame, (50, 150 + int(250 * (1 - angle_percentage))), (85, 400), color, cv2.FILLED)

		return frame


	def process_image(self, frame):
		landmarks, results = self.pose_detector.get_landmark_positions(frame)
		frame = self.pose_detector.draw_landmarks(frame, results, None)
		frame = self.draw_joint_circles(frame, landmarks)

		if self.is_press_up_position(landmarks, frame) == True:
			averages = []
			for id1, id2 in [[15, 16], [13, 14], [11, 12]]:
				average_y = (landmarks[id1][1] + landmarks[id2][1]) / 2
				average_x = (landmarks[id1][0] + landmarks[id2][0]) / 2

				averages.append([average_y, average_x])

			angle = self.calculate_angle(*averages[0], *averages[1], *averages[2])


			#normalize angle
			if angle > self.max_angle:
				angle = self.max_angle
			elif angle < self.min_angle:
				angle = self.min_angle


			#check if new repetition has been completed
			if angle <= self.min_angle:
				if not self.has_dropped:
					self.has_dropped = True
					self.color = self.up_color

			if angle >= self.max_angle:
				if self.has_dropped:
					self.repetitions += 1
					self.has_dropped = False
					self.color = self.down_color


			frame = self.draw_progress_bar(frame, angle, self.color)
		else:
			cv2.putText(frame, 'Resting...', (600, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
			self.has_dropped = False


		cv2.putText(frame, f'Repetitions: {self.repetitions}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (257, 45, 0), 2)


		return frame




if __name__ == '__main__':
	counter = exercise_counter()
	run = True 


	while run:
		status, frame = counter.take_picture()

		if status:
			frame = counter.process_image(frame)
			cv2.imshow('Exercise Counter', frame)

		if cv2.waitKey(1) & 0xFF == ord('q') or not status:
			run = False