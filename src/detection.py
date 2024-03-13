import sys
sys.path.append('/content/TennisProject/src')
import operator
import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from scipy import signal

from court_detection import CourtDetector
from sort import Sort
from utils import get_video_properties, get_dtype
import matplotlib.pyplot as plt


class DetectionModel:
    def __init__(self, dtype=torch.FloatTensor):
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.type(dtype)  # Also moves model to GPU if available
        self.detection_model.eval()
        self.dtype = dtype
        self.PERSON_LABEL = 1
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.85
        self.PERSON_SECONDARY_SCORE = 0.3
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.persons_boxes = {}
        self.persons_dists = {}
        self.persons_first_appearance = {}
        self.counter = 0
        self.num_of_misses = 0
        self.last_frame = None
        self.current_frame = None
        self.next_frame = None
        self.movement_threshold = 200
        self.mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.05)

    def detect_player_1(self, image, court_detector):
        """
        Detecting bottom player
        """
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        # Check if player has been detected before
        if len(self.player_1_boxes) == 0:
            if court_detector is None:
                image_court = image.copy()
            else:
                # ROI is bottom half of the court
                court_type = 1
                white_ref = court_detector.court_reference.get_court_mask(court_type)
                white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], image.shape[1::-1])

                image_court = image.copy()
                image_court[white_mask == 0, :] = (0, 0, 0)

            # Detect all persons in the ROI
            persons_boxes, _ = self._detect(image_court)
            if len(persons_boxes) > 0:
                # Choose person with biggest box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                biggest_box = max(persons_boxes, key=lambda x: area_of_box(x)).round()
                self.player_1_boxes.append(biggest_box)
            else:
                return None
        else:
            # Use previous player location to define new ROI
            xt, yt, xb, yb = self.player_1_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]

            # Detect all person in ROI
            persons_boxes, _ = self._detect(trimmed_image, self.PERSON_SECONDARY_SCORE)
            if len(persons_boxes) > 0:
                # Find person closest to previous detection
                c1 = center_of_box(self.player_1_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (
                    box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_1_boxes.append(closest_box)
                else:
                    # Counter is to decide if box has not been found for more than number of frames
                    self.counter += 1
                    self.player_1_boxes.append(self.player_1_boxes[-1])
            else:
                self.player_1_boxes.append(self.player_1_boxes[-1])
                self.num_of_misses += 1
        cv2.rectangle(boxes, (int(self.player_1_boxes[-1][0]), int(self.player_1_boxes[-1][1])),
                      (int(self.player_1_boxes[-1][2]), int(self.player_1_boxes[-1][3])), [255, 0, 255], 2)

        return boxes

    def detect_top_persons(self, image, court_detector, frame_num):
        """
        Detect all persons in the top half of the court
        """
        boxes = np.zeros_like(image)

        if court_detector is None:
            image_court = image.copy()
        else:
            # Define ROI to be top half of the court
            court_type = 2
            white_ref = court_detector.court_reference.get_court_mask(court_type)
            white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], image.shape[1::-1])
            white_mask = cv2.dilate(white_mask, np.ones((100, 1)), anchor=(0, 0))
            image_court = image.copy()
            image_court[white_mask == 0, :] = (0, 0, 0)

        # Detect all the persons in the top half court
        persons_boxes, probs = self._detect(image_court, self.PERSON_SECONDARY_SCORE)
        if len(persons_boxes) == 0:
            persons_boxes, probs = None, None

        # Track persons using SORT algorithm
        tracked_objects = self.mot_tracker.update(persons_boxes, probs)
        for det_person in self.persons_boxes.keys():
            self.persons_boxes[det_person].append([None, None, None, None])
        # Mark each person box
        for box in tracked_objects:
            cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
            cv2.putText(boxes, f'Player {int(box[4])}', (int(box[0]) - 10, int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if int(box[4]) in self.persons_boxes.keys():
                self.persons_boxes[int(box[4])][-1] = box[:4]
            else:
                self.persons_boxes[int(box[4])] = [box[:4]]
                self.persons_first_appearance[int(box[4])] = frame_num

        return boxes

    def calculate_all_persons_dists(self):
        """
        For each person detected in top half court, calculate the distance their box has moved in the video
        """
        for id, person_boxes in self.persons_boxes.items():
            person_boxes = [box for box in person_boxes if box[0] is not None]
            dist = boxes_dist(person_boxes)
            self.persons_dists[id] = dist
        return self.persons_dists

    def find_player_2_box(self):
        """
        Discriminate the top player box from other persons in the frame
        """
        dists = self.calculate_all_persons_dists()
        min_length = 20
        boxes = []

        # Get frames section for each person detected
        persons_sections = {key: [val, val + len(self.persons_boxes[key]) - np.argmax([box[0] != None for box in self.persons_boxes[key][::-1]]) - 1] for key, val in
                            self.persons_first_appearance.items()}
        detections = []
        # Find max dist box for each iteration and remove section that intersect with max dist box`s section
        while len(dists) > 0:
            max_key = max(dists.items(), key=operator.itemgetter(1))[0]
            max_sec = persons_sections[max_key].copy()
            if max_sec[1] - max_sec[0] < min_length:
                dists.pop(max_key)

                continue
            detections.append(max_key)
            dists.pop(max_key)
            for key, sec in persons_sections.items():
                if sections_intersect(max_sec, sec):
                    if key in dists.keys():
                        dists.pop(key)
        detections = sorted(detections)

        # Combine all detection to one person boxes
        for det in detections:
            start = self.persons_first_appearance[det]
            missing = start - 1 - len(boxes)
            boxes.extend([[None, None, None, None]] * missing)
            boxes.extend(self.persons_boxes[det][:persons_sections[det][1] - persons_sections[det][0] + 1])
        missing = len(self.player_1_boxes) - len(boxes)
        boxes.extend([[None, None, None, None]] * missing)
        self.player_2_boxes = boxes

    def _detect(self, image, person_min_score=None):
        """
        Use deep learning model to detect all person in the image
        """
        if person_min_score is None:
            person_min_score = self.PERSON_SCORE_MIN
        # creating torch.tensor from the image ndarray
        frame_t = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.detection_model(frame_tensor)

        persons_boxes = []
        probs = []
        for box, label, score in zip(p[0]['boxes'][:], p[0]['labels'], p[0]['scores']):
            if label == self.PERSON_LABEL and score > person_min_score:
                '''cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(boxes, 'Person %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)'''
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(score.detach().cpu().numpy())
        return persons_boxes, probs

    def calculate_feet_positions(self, court_detector):
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes
        of both players
        """
        inv_mats = court_detector.game_warp_matrix
        positions_1 = []
        positions_2 = []
        # Bottom player feet locations
        for i, box in enumerate(self.player_1_boxes):
            feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()]).reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
            positions_1.append(feet_court_pos)
        mask = []
        # Top player feet locations
        for i, box in enumerate(self.player_2_boxes):
            if box[0] is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_2.append(feet_court_pos)
                mask.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask.append(False)

        # Smooth both feet locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)

        smoothed_2[not mask, :] = [None, None]
        return smoothed_1, smoothed_2


def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2


def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width


def boxes_dist(boxes):
    """
    Calculate the cumulative distance of all the boxes
    """
    total_dist = 0
    for box1, box2 in zip(boxes, boxes[1:]):
        box1_center = np.array(center_of_box(box1))
        box2_center = np.array(center_of_box(box2))
        dist = np.linalg.norm(box2_center - box1_center)
        total_dist += dist
    return total_dist


def sections_intersect(sec1, sec2):
    """
    Check if two sections intersect
    """
    if sec1[0] <= sec2[0] <= sec1[1] or sec2[0] <= sec1[0] <= sec2[1]:
        return True
    return False


if __name__ == "__main__":
    court_detector = CourtDetector()
    video = cv2.VideoCapture('../videos/vid26.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output videos writer
    out = cv2.VideoWriter(os.path.join('output', 'player2test.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))
    dtype, device = get_dtype()
    model = DetectionModel(dtype)
    frame_i = 0
    while True:
        ret, frame = video.read()
        frame_i += 1
        if ret:
            if frame_i == 1:
                court_detector.detect(frame)
            court_detector.track_court(frame)

            frame = model.detect_top_persons(frame, court_detector, frame_i)
            out.write(frame)
            cv2.imshow('df', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        else:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()


    dists = model.calculate_all_persons_dists()
    threshold = 1
    to_del = []
    for key, val in dists.items():
        length = len(model.persons_boxes[key])
        if length > 0:
            average_dist = val / length
        else:
            average_dist = 0
        if average_dist < threshold:
            to_del.append(key)
    for key in to_del:
        dists.pop(key)
        model.persons_first_appearance.pop(key)
    print(dists)
    print(model.persons_first_appearance)
    persons_sections = {key: [val, val + len(model.persons_boxes[key])] for key, val in model.persons_first_appearance.items()}
    print(persons_sections)
    detections = []
    while len(dists) > 0:
        max_key = max(dists.items(), key=operator.itemgetter(1))[0]
        max_sec = persons_sections[max_key].copy()
        detections.append(max_key)
        persons_sections.pop(max_key)
        dists.pop(max_key)
        for key, sec in persons_sections.items():
            if sections_intersect(max_sec, sec):
                if key in dists.keys():
                    dists.pop(key)
    detections = sorted(detections)
    print(detections)
    boxes = []
    for person in detections:
        start = model.persons_first_appearance[person]
        missing = start - 1 - len(boxes)
        boxes.extend([None] * missing)
        boxes.extend(model.persons_boxes[person])

    video = cv2.VideoCapture('../videos/vid26.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output videos writer
    out = cv2.VideoWriter(os.path.join('output', 'player2test.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))


    frame_i = 0
    while True:
        ret, frame = video.read()
        frame_i += 1
        if ret:

            if frame_i <= len(boxes):
                box = boxes[frame_i - 1]
            else:
                box = None
            if box is not None:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
            out.write(frame)
            cv2.imshow('df', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        else:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()

    #model.find_player_2_box()
