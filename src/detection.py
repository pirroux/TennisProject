import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd

from src.court_detection import CourtDetector
from src.sort import Sort
from src.utils import get_video_properties, get_dtype
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
        self.PERSON_SECONDARY_SCORE = 0.5
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.persons_boxes = []
        self.counter = 0
        self.im_diff = ImageDiff()
        self.backSub = cv2.createBackgroundSubtractorKNN()
        self.num_of_misses = 0
        self.last_frame = None
        self.current_frame = None
        self.next_frame = None
        self.movement_threshold = 200
        self.mot_tracker = Sort()

    def detect_player_1(self, image, court_detector):
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        if len(self.player_1_boxes) == 0:
            if court_detector is None:
                image_court = image.copy()
            else:
                court_type = 1
                white_ref = court_detector.court_reference.get_court_mask(court_type)
                white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], image.shape[1::-1])
                # TODO find different way to add more space at the top
                if court_type == 2:
                    white_mask = cv2.dilate(white_mask, np.ones((50, 1)), anchor=(0, 0))
                image_court = image.copy()
                image_court[white_mask == 0, :] = (0, 0, 0)
            '''max_values = np.max(np.max(image_court, axis=1), axis=1)
            max_values_index = np.where(max_values > 0)[0]
            top_y = max_values_index[0]
            bottom_y = max_values_index[-1]'''
            # cv2.imwrite('../report/frame_only_court.png', image_court)
            '''cv2.imshow('res', image_court)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            # mask = self.find_canadicate(image)
            # image[mask == 0, :] = (0,0,0)
            # image_court = image_court[top_y:bottom_y, :, :]

            persons_boxes, _ = self._detect(image_court)
            if len(persons_boxes) > 0:
                # TODO find a different way to choose correct box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                biggest_box = max(persons_boxes, key=lambda x: area_of_box(x)).round()
                self.player_1_boxes.append(biggest_box)
            else:
                return None
        else:
            xt, yt, xb, yb = self.player_1_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]
            '''cv2.imshow('res', trimmed_image)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            persons_boxes, _ = self._detect(trimmed_image, self.PERSON_SECONDARY_SCORE)
            if len(persons_boxes) > 0:
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
                # TODO the patch is small so this might not be needed
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

    def detect_top_persons(self, image, court_detector):
        frame = image.copy()

        if court_detector is None:
            image_court = image.copy()
        else:
            court_type = 2
            white_ref = court_detector.court_reference.get_court_mask(court_type)
            white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], image.shape[1::-1])
            white_mask = cv2.dilate(white_mask, np.ones((100, 1)), anchor=(0, 0))
            image_court = image.copy()
            image_court[white_mask == 0, :] = (0, 0, 0)

        # cv2.imwrite('../report/frame_only_court.png', image_court)
        '''cv2.imshow('res', image_court)
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()'''

        persons_boxes, probs = self._detect(image_court, self.PERSON_SECONDARY_SCORE)
        if len(persons_boxes) > 0:
            self.persons_boxes.append(persons_boxes)

        else:
            persons_boxes, probs = np.empty((0, 4)), [0]
            self.persons_boxes.append([])

        tracked_objects = self.mot_tracker.update(persons_boxes, probs)
        for box in tracked_objects:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
            cv2.putText(frame, f'Player {int(box[4])}', (int(box[0]) - 10, int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return frame

    def find_player_2_box(self):
        boxes_centers = []
        for boxes in self.persons_boxes:
            if boxes is not None:
                centers = [center_of_box(box) for box in boxes]
                boxes_centers.append(centers)
            else:
                boxes_centers.append([])

        boxes_centers = np.array(boxes_centers, dtype=object)
        max_len = len(max(boxes_centers, key=lambda x: len(x)))
        list1 = [[] for _ in range(max_len)]
        for frame in boxes_centers:
            for i, cen in enumerate(frame):
                list1[i].append(cen)
            for j in range(len(frame), max_len):
                list1[j].append((None, None))
        list1 = np.array(list1)
        plt.figure()
        for list in list1:
            x_values = list[:, 0]
            plt.scatter(range(len(list)), x_values, c='b')
        plt.show()

    def _detect(self, image, person_min_score=None):
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

    def diff_image(self, next_frame):
        gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        self.last_frame = self.current_frame
        self.current_frame = self.next_frame
        self.next_frame = gray.copy()
        if self.last_frame is not None:
            first_motion = abs(self.current_frame - self.last_frame)
            first_motion = cv2.threshold(first_motion, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
            second_motion = abs(self.next_frame - self.last_frame)
            second_motion = cv2.threshold(second_motion, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
            motion_matrix = first_motion.copy()
            motion_matrix[second_motion > 0] = 0

            motion_matrix = cv2.dilate(motion_matrix, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))

            contours, _ = cv2.findContours(motion_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            f = cv2.cvtColor(motion_matrix, cv2.COLOR_GRAY2BGR)
            fr = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
            z = np.zeros_like(f)
            for i, c in enumerate(contours):
                if 2000 < cv2.contourArea(c) < 10000:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.drawContours(z, contours_poly, i, (255, 0, 0))
            cv2.imshow('sdf', fr)
            if cv2.waitKey(100) & 0xff == 27:
                cv2.destroyAllWindows()

    def find_canadicate(self, image):
        frame = image.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 3)

        fgMask = self.backSub.apply(frame)
        fgMask = cv2.threshold(fgMask, 10, 1, cv2.THRESH_BINARY)[1]

        diff = self.im_diff.diff(frame)

        res = diff * fgMask * 255
        res = cv2.dilate(res, np.ones((40, 25)))
        contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = []
        boundRects = []
        max_area = 0
        max_c = None
        for c in contours:
            if 200 < cv2.contourArea(c) < 12000:
                contours_poly.append(cv2.approxPolyDP(c, 3, True))
                boundRects.append(cv2.boundingRect(contours_poly[-1]))

        drawing = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)
        f = image.copy()
        mask = np.ones_like(image)
        for i, boundRect in enumerate(boundRects):
            mask[int(boundRect[1]):int(boundRect[1] + boundRect[3]), int(boundRect[0]):int(boundRect[0] + boundRect[2]),
            :] = (0, 0, 0)
            f = f * mask
            '''cv2.imshow('res', box)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            color = (0, 0, 255)
            cv2.drawContours(drawing, contours_poly, i, (255, 0, 0))
            cv2.rectangle(drawing, (int(boundRect[0]), int(boundRect[1])),
                          (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), color, 2)

        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        side_by_side = np.concatenate([res, image], axis=1)
        side_by_side = cv2.resize(side_by_side, (1920, 540))
        cv2.imshow('Contours', side_by_side)
        c = image.copy()
        c[res == 0] = 0

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        return res


def center_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2


def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width


def boxes_diff(frame, box1, box2):
    box1 = frame[box1[3]:box1[1], box1[2]:box1[0]].copy()
    box2 = frame[box2[3]:box2[1], box2[2]:box2[0]].copy()
    avg_color1 = np.mean(np.mean(box1, axis=0), axis=0)


class ImageDiff:
    def __init__(self):
        self.last_image = None
        self.diff_image = None

    def diff(self, image):
        if self.last_image is None:
            self.last_image = image.copy()
            return np.ones_like(image)
        else:
            self.diff_image = abs(self.last_image - image)
            self.diff_image = cv2.threshold(self.diff_image, 200, 1, cv2.THRESH_BINARY)[1]
            return self.diff_image


if __name__ == "__main__":
    '''centers = [[(465.0529327392578, 78.30905723571777), (513.7422637939453, 112.076416015625)], [(465.04197692871094, 78.31762313842773), (513.8127136230469, 111.96683120727539)], [(465.0560302734375, 78.2927131652832), (513.8919982910156, 111.94735717773438)], [(512.8419189453125, 111.11042022705078), (465.0616912841797, 78.30221366882324)], [(512.8310699462891, 111.10095977783203), (465.057373046875, 78.341796875)], [(465.25213623046875, 78.29021644592285), (513.3552093505859, 111.72183990478516)], [(466.4238586425781, 78.13817024230957), (513.8712768554688, 111.56658554077148)], [(512.498779296875, 109.28411865234375), (466.4071044921875, 78.07453346252441)], [(512.7337799072266, 110.56003189086914), (466.4268035888672, 78.06708145141602)], [(512.4842987060547, 110.85307312011719), (465.5473175048828, 78.23431777954102)], [(513.7494659423828, 111.99290084838867), (465.43235778808594, 78.38398170471191)], [(512.6632537841797, 110.96553421020508), (466.4927673339844, 78.23144721984863)], [(513.7677001953125, 111.8534164428711), (466.5242614746094, 78.2228946685791)], [(513.7413940429688, 111.78302383422852), (466.5677947998047, 78.07649230957031)], [(513.4419097900391, 111.82112884521484), (465.66700744628906, 78.27528762817383), (286.2524719238281, 282.00804138183594)], [(512.8977203369141, 109.65666580200195), (465.6986999511719, 78.29911994934082), (286.2826385498047, 281.95391845703125)], [(512.8923187255859, 109.79841995239258), (466.7819366455078, 78.09141540527344), (286.2940673828125, 281.8891143798828)], [(513.0466918945312, 109.67992782592773), (466.7858581542969, 78.15314483642578), (286.2782745361328, 281.82542419433594)], [(513.1065979003906, 109.5023193359375), (466.830322265625, 78.05434036254883), (286.3241424560547, 281.91334533691406)], [(513.0148773193359, 109.57709884643555), (465.4287567138672, 78.77929306030273), (286.28797912597656, 281.67132568359375)], [(513.1466979980469, 110.45702362060547), (466.77728271484375, 78.32157707214355), (286.3146209716797, 281.67247009277344)], [(513.0781860351562, 110.4638671875), (466.86289978027344, 78.09329795837402), (286.37091064453125, 281.70494079589844)], [(513.0482635498047, 110.92447280883789), (466.85748291015625, 78.1185302734375), (286.3522186279297, 281.85418701171875)], [(513.3861083984375, 111.06557846069336), (466.81085205078125, 78.09846878051758), (286.3699493408203, 282.0002136230469)], [(513.1264190673828, 109.98699188232422), (466.6840057373047, 78.0782299041748), (286.27210998535156, 282.44239807128906)], [(513.0141754150391, 110.45398330688477), (466.6163635253906, 77.9874210357666), (286.3611297607422, 282.70457458496094)], [(513.9755096435547, 111.18378448486328), (465.39561462402344, 78.27710342407227), (286.356201171875, 282.64903259277344)], [(513.9022827148438, 111.20816421508789), (465.26373291015625, 78.19904708862305), (286.2420654296875, 282.69805908203125)], [(513.7148895263672, 110.6264762878418), (465.2480926513672, 78.22270774841309), (286.28282165527344, 282.81683349609375)], [(514.2623596191406, 109.4631576538086), (465.3348388671875, 78.2904281616211), (286.4091033935547, 282.1127471923828)], [(513.8701477050781, 108.28432846069336), (465.2951965332031, 77.78035926818848)], [(514.0196075439453, 107.74723434448242), (465.5541687011719, 77.91241455078125)], [(514.2791137695312, 106.9882926940918), (465.5632019042969, 77.92840766906738)], [(514.2800140380859, 106.98513793945312), (465.4910583496094, 77.9412784576416)], [(514.2813415527344, 106.99088668823242), (465.43133544921875, 78.01067161560059)], [(514.8207397460938, 106.15756225585938), (466.03880310058594, 78.28113746643066), (286.707275390625, 282.2636260986328)], [(515.1423645019531, 105.81194305419922), (466.4616394042969, 77.9493637084961)], [(515.4971771240234, 105.78910446166992), (464.77720642089844, 77.77583122253418)], [(516.0649108886719, 106.28768920898438), (465.8715057373047, 78.05184936523438)], [(516.1411743164062, 107.03393936157227), (466.35227966308594, 78.14776802062988)], [(516.8159790039062, 108.16672134399414), (464.8335266113281, 77.953857421875)], [(518.1406097412109, 109.69424438476562), (466.1975402832031, 78.13199615478516)], [(518.4118499755859, 111.66580963134766), (465.8813171386719, 77.18462562561035)], [(519.7083129882812, 113.28976440429688), (465.25732421875, 76.49195861816406)], [(520.1200866699219, 114.4957275390625), (465.5735321044922, 76.7651309967041)], [(519.9415740966797, 116.05774688720703), (464.8692169189453, 75.6718921661377)], [(520.3475952148438, 118.80271530151367), (464.30517578125, 75.5008716583252)], [(520.1611022949219, 120.10213088989258), (464.3008270263672, 75.37334060668945)], [(520.3892669677734, 119.36908340454102), (464.7104797363281, 75.69867706298828)], [(521.2786712646484, 119.59498596191406), (464.3847351074219, 75.53935432434082)], [(522.9123077392578, 119.19538879394531), (464.67823791503906, 75.64436149597168)], [(464.74388122558594, 75.64212226867676), (523.7505187988281, 117.76251983642578)], [(525.0314788818359, 117.85784149169922), (464.1114501953125, 75.34895324707031)], [(526.8006439208984, 117.10773849487305), (464.1236114501953, 75.35713195800781)], [(529.1318359375, 116.02252578735352), (464.6267547607422, 75.52108383178711)], [(531.5516204833984, 115.54145812988281), (464.5458526611328, 75.45429420471191)], [(534.1390533447266, 115.0948600769043), (464.5599822998047, 75.46504211425781)], [(535.9940490722656, 114.07161712646484), (464.5646667480469, 75.46272659301758)], [(538.3219299316406, 111.84677505493164), (464.8505096435547, 76.68811225891113)], [(541.9733581542969, 109.95159149169922), (464.3266296386719, 75.62560272216797)], [(546.20556640625, 109.25655746459961), (465.1184997558594, 76.78729057312012)], [(553.8323364257812, 109.60224151611328), (465.0767822265625, 76.90898895263672)], [(553.8323364257812, 109.60224151611328), (465.0767822265625, 76.90898895263672)], [(551.3174743652344, 110.30897521972656), (464.9671173095703, 76.47844696044922)], [(549.2991333007812, 109.85100173950195), (465.4674530029297, 76.06550025939941)], [(549.2991333007812, 109.85100173950195), (465.4674530029297, 76.06550025939941)], [(547.2206726074219, 112.14334869384766), (465.3975372314453, 76.07165336608887)], [(546.7305908203125, 112.90623474121094), (464.91607666015625, 75.25659370422363)], [(546.7305908203125, 112.90623474121094), (464.91607666015625, 75.25659370422363)], [(465.3151397705078, 75.03961944580078), (547.2873229980469, 113.78353881835938)], [(465.3151397705078, 75.03961944580078), (547.2873229980469, 113.78353881835938)], [(466.86219787597656, 76.11861801147461), (548.7099914550781, 114.47311782836914)], [(467.13836669921875, 76.5069408416748), (552.7558898925781, 118.5206069946289)], [(467.1409149169922, 76.50695610046387), (552.7583618164062, 118.51393127441406)], [(555.359619140625, 117.6800422668457), (468.45904541015625, 76.889892578125)], [(559.0533447265625, 121.16157913208008), (471.34130859375, 76.68834114074707)], [(559.0337219238281, 121.1625862121582), (471.3435516357422, 76.69876861572266)], [(559.0807189941406, 122.62170791625977), (473.60028076171875, 77.4977798461914)], [(561.2861633300781, 123.89185333251953)], [(561.2860412597656, 123.89187240600586)], [(566.7190551757812, 123.72108840942383)], [(576.2445678710938, 124.57087707519531)], [(582.6217346191406, 124.62035369873047)], [(585.5863952636719, 125.08512115478516)], [(588.9769287109375, 124.71322250366211)], [(591.0679626464844, 124.1855697631836), (595.4927673339844, 111.16057205200195)], [(593.8447875976562, 123.5133171081543)], [(598.4639587402344, 121.99127960205078)], [(604.1161193847656, 120.89380645751953)], [(607.5943603515625, 119.28003311157227)], [(613.1956787109375, 117.93386840820312)], [(617.0408630371094, 117.37080764770508)], [(622.7530822753906, 116.5495719909668)], [(628.2067565917969, 115.29545974731445)], [(290.4903106689453, 280.64874267578125), (633.7725524902344, 114.09114456176758)], [(638.5438842773438, 114.10176086425781)], [(642.4549255371094, 113.01122283935547)], [(645.6065368652344, 111.42046737670898)], [(648.5676879882812, 111.62367630004883)], [(653.1451721191406, 110.47379684448242), (665.0373229980469, 100.98835372924805), (669.9696655273438, 75.91864395141602)], [(666.8113098144531, 101.06718826293945), (657.044189453125, 107.7635726928711), (669.7619323730469, 76.75986099243164)], [(668.7559814453125, 101.41059112548828)], [(671.983642578125, 100.66404342651367)], [(673.2835998535156, 105.09586334228516)], [(676.2096862792969, 100.59634017944336)], [(679.7954711914062, 101.6419792175293)], [(684.1201782226562, 100.26635360717773), (297.41468811035156, 229.38015747070312)], [(691.1408081054688, 101.69370651245117)], [(703.6968078613281, 105.40961456298828), (680.6639099121094, 75.79287147521973)], [(704.7553100585938, 103.76226043701172), (304.71270751953125, 228.0835723876953)], [(711.8254089355469, 105.9789810180664), (686.6965637207031, 77.65700149536133)], [(717.5323181152344, 107.64639663696289), (691.1661376953125, 76.7877082824707)], [(718.5658569335938, 107.93350601196289), (694.3753967285156, 73.54159355163574)], [(721.1622619628906, 107.6851806640625)], [(722.9859313964844, 107.52108764648438)], [(726.6561584472656, 106.00596237182617)], [(730.4699096679688, 104.69234848022461)], [(733.4709777832031, 102.35726547241211)], [(736.2142639160156, 101.45083618164062)], [(738.1211547851562, 100.69513702392578)], [(741.8210754394531, 99.25190353393555)], [(743.6120910644531, 98.90907287597656)], [(746.6176147460938, 98.56256103515625)], [(748.3069152832031, 98.36594772338867)], [(750.5754699707031, 99.93743515014648)], [(751.8724365234375, 100.74197769165039)], [(753.4765930175781, 102.77402877807617)], [(753.6611022949219, 103.64427185058594)], [(754.4845886230469, 104.4985122680664)], [(754.5851135253906, 103.57042694091797)], [(756.2185974121094, 104.61133193969727)], [(756.0305786132812, 103.9844970703125), (716.9084167480469, 70.10453796386719)], [(757.4480895996094, 104.00457382202148), (716.998046875, 72.20423316955566)], [(760.2803955078125, 103.86064147949219), (716.9670104980469, 72.31063270568848)], [(767.2703552246094, 103.14577102661133)], [(773.1524963378906, 101.74011611938477)], [(780.7315979003906, 102.52162551879883)], [(787.3856506347656, 102.14531326293945)], [(788.3157043457031, 102.55513763427734)], [(790.3441467285156, 102.12105560302734)], [(792.3566284179688, 101.31632614135742)], [(795.6510925292969, 101.38055038452148), (805.3320007324219, 84.3763427734375)], [(806.1802062988281, 101.80064010620117)], [(811.1510925292969, 101.37889862060547)], [(815.09765625, 101.90830993652344)], [(820.4014587402344, 101.64192962646484)], [(826.953125, 99.68321990966797)], [(832.1239318847656, 99.87123107910156)], [(836.7633056640625, 100.3455924987793), (870.9507751464844, 74.85058784484863)], [(837.9649963378906, 99.71811294555664), (872.8074035644531, 74.42075729370117)], [(852.8995971679688, 96.88104248046875)], [(855.4579162597656, 96.73360824584961)], [(855.6985473632812, 96.96061706542969)], [(859.1130065917969, 97.01552200317383)], [(861.6375122070312, 97.00590133666992)], [(864.9852905273438, 98.09685134887695)], [(869.9977722167969, 99.90636825561523)], [(876.056396484375, 98.97070693969727)], [(878.3547058105469, 102.84309005737305)], [(881.4028625488281, 105.54361343383789)], [(885.0555725097656, 104.3108901977539)], [(885.74169921875, 105.58453750610352)], [(889.3470153808594, 106.58625030517578)], [(888.5320434570312, 106.9851303100586)], [(895.760498046875, 104.45810317993164)], [(899.4631652832031, 103.12907791137695)], [(905.2109985351562, 104.32194519042969), (874.7221984863281, 76.15580368041992)], [(906.8470458984375, 104.69289016723633), (874.2611389160156, 74.77111434936523)], [(908.951904296875, 105.44253540039062)], [(911.7291259765625, 106.58995819091797), (872.1129455566406, 74.39735412597656)], [(913.6106872558594, 106.54276275634766)], [(912.2691345214844, 105.71195983886719)], [(910.8822326660156, 106.03654861450195)], [(909.5595397949219, 105.98785018920898)], [(906.6842041015625, 105.00593948364258)], [(906.12353515625, 104.53532028198242)], [(906.9277954101562, 102.92509078979492)], [(906.5250244140625, 102.1157455444336)], [(907.3083190917969, 100.31571960449219)], [(906.0933837890625, 100.08344268798828)], [(904.3923645019531, 99.8766860961914)], [(901.1647338867188, 99.08605194091797)], [(897.9466552734375, 98.43739700317383)], [(891.9198608398438, 97.73892593383789)], [(887.8206481933594, 97.13501358032227)], [(883.4516296386719, 96.25590515136719)], [(881.7882995605469, 96.76916885375977)], [(879.6165771484375, 96.42271423339844)], [(876.598388671875, 96.03832626342773)], [(873.9100036621094, 96.93914794921875)], [(870.4669494628906, 96.0234489440918)], [(866.886962890625, 97.13431930541992), (879.4244079589844, 75.95431137084961)], [(860.583740234375, 96.27392959594727), (880.471435546875, 77.18524360656738)], [(855.3991394042969, 96.05287551879883), (883.3980407714844, 74.84959030151367), (876.1972045898438, 74.80653190612793)], [(850.87548828125, 95.7799301147461), (876.7713623046875, 74.99576187133789), (884.0213623046875, 74.84195518493652)], [(845.9180297851562, 97.33082962036133), (875.2826843261719, 74.76083946228027), (884.2113037109375, 74.91316032409668)], [(842.8664855957031, 96.9214973449707), (874.3129577636719, 74.89030838012695)], [(838.6956176757812, 97.33807754516602), (874.903076171875, 74.66962242126465), (884.7408447265625, 75.12295150756836)], [(835.4355773925781, 96.85969161987305), (874.8198852539062, 74.25025749206543), (883.1324462890625, 74.19575881958008)], [(833.3936462402344, 97.8322525024414), (875.6119384765625, 74.47626876831055), (883.6754455566406, 74.38424682617188)], [(830.9956359863281, 96.90088272094727), (875.3281555175781, 74.47307014465332)], [(828.2069396972656, 96.88422775268555), (875.4575500488281, 74.5118522644043)], [(826.3922729492188, 96.603759765625), (874.4424438476562, 74.88773345947266)], [(823.900146484375, 96.47029495239258), (873.8935852050781, 75.24218559265137)], [(822.0870666503906, 96.34656524658203), (873.8516845703125, 75.48277854919434)], [(820.8985900878906, 95.71396255493164), (873.7988891601562, 75.55996131896973)], [(819.8704223632812, 94.93945693969727), (874.5327758789062, 75.0039291381836)], [(817.3672485351562, 95.40196228027344), (874.6166381835938, 74.90816688537598)], [(815.7296447753906, 95.38802719116211), (874.2982788085938, 75.14450645446777)], [(813.4709777832031, 95.19894027709961), (873.7170104980469, 75.20059204101562)], [(810.6520385742188, 94.79909896850586)], [(807.6946411132812, 95.39681625366211)], [(805.3462219238281, 95.80405807495117)], [(803.7138977050781, 96.5447006225586)], [(803.3401489257812, 97.10139846801758)], [(803.2167663574219, 97.94809341430664)], [(802.9508361816406, 98.85994338989258)], [(803.9332885742188, 98.30428314208984)], [(803.5884094238281, 98.21355056762695)], [(799.9964904785156, 97.64452743530273)], [(794.8576965332031, 98.07648849487305)], [(786.8988037109375, 96.62045669555664)], [(781.9609680175781, 98.42922973632812)], [(776.017578125, 98.89244842529297)], [(773.6961364746094, 99.5999984741211)], [(771.7569580078125, 98.57376098632812), (718.7883911132812, 75.40094375610352)], [(768.5833435058594, 98.95430374145508), (717.7623596191406, 76.16031646728516)], [(764.2987060546875, 99.03822326660156), (717.2414855957031, 75.95412063598633)], [(757.4532470703125, 99.12181091308594), (715.5400390625, 75.96991729736328)], [(746.1216125488281, 99.35976028442383), (715.5973815917969, 75.57075119018555)], [(740.9181518554688, 99.18921661376953), (715.0175170898438, 75.79934310913086)], [(729.9716491699219, 97.57075119018555)], [(727.6037292480469, 97.14527130126953)], [(725.6091918945312, 96.52516174316406)], [(720.1779174804688, 95.75689315795898)], [(713.269775390625, 94.18618774414062)], [(708.8188781738281, 94.90932083129883)], [(706.2088012695312, 95.67910385131836), (715.512939453125, 80.07939720153809)], [(701.7146911621094, 94.53407669067383), (713.9212646484375, 81.26477813720703)], [(696.2416687011719, 95.728759765625), (718.8054504394531, 75.76908111572266), (709.4165649414062, 86.41570091247559)], [(690.0637817382812, 95.57288360595703), (715.5233459472656, 70.41131019592285)], [(683.956298828125, 94.89347076416016), (714.7655639648438, 72.62529945373535)], [(676.1615600585938, 93.83363151550293), (714.5516662597656, 74.74180603027344)], [(671.3167114257812, 93.83281707763672), (713.1978454589844, 75.25946426391602)], [(658.3715209960938, 94.69926071166992), (713.6466979980469, 75.26993179321289)], [(653.5574340820312, 95.22745513916016), (713.3279724121094, 75.28707504272461)], [(647.92041015625, 94.85533905029297)], [(641.8081359863281, 95.78636932373047)], [(634.2005920410156, 93.72130012512207)], [(627.383544921875, 92.97074317932129)], [(622.4022521972656, 92.4455451965332)], [(616.7171020507812, 90.62630844116211)], [(610.4503479003906, 90.64961624145508)], [(603.3851623535156, 90.9033432006836)], [(598.7090148925781, 90.0511703491211)], [(592.4399108886719, 90.7476577758789)], [(586.9715881347656, 91.6838493347168)], [(582.9504699707031, 91.67364883422852)], [(579.3735046386719, 91.62996673583984), (566.3282470703125, 73.24615287780762)], [(576.3996276855469, 91.23299217224121)], [(573.0224304199219, 90.38940620422363), (550.871826171875, 75.18071937561035)], [(570.3924865722656, 90.52009963989258), (549.1686096191406, 80.29600524902344)], [(568.0599060058594, 90.03198432922363)], [(567.710693359375, 90.3242130279541)], [(566.3610229492188, 89.2656364440918)], [(565.2764892578125, 90.44474220275879)], [(563.9649047851562, 91.42301559448242)], [(566.2684631347656, 91.12711334228516)], [(565.0045776367188, 90.51585388183594)], [(565.7780456542969, 91.62276458740234), (577.6812133789062, 87.12403869628906)], [(566.546630859375, 91.83347702026367)], [(567.4712219238281, 92.19038772583008)], [(569.4338989257812, 93.03571701049805)], [(571.0040893554688, 93.58596420288086)], [(571.3691711425781, 93.57090759277344)], [(573.9531555175781, 91.91560363769531)], [(574.9174194335938, 90.84011459350586)], [], [], [], [(579.2597961425781, 90.70840454101562)], [(581.232421875, 91.20811653137207)], [(582.5817565917969, 92.50150299072266)], [(584.5425109863281, 92.65703964233398)], [(588.019287109375, 91.60804557800293)], [(590.3079833984375, 91.54965019226074)], [(591.3825073242188, 92.66618919372559)], [(594.9544677734375, 91.48478507995605)], [(595.1190795898438, 92.54795455932617)], [(597.2365417480469, 92.21306610107422)], [(598.1025695800781, 91.9090347290039)], [(597.7676086425781, 91.21289443969727)], [(599.4429321289062, 90.92095565795898)], [(600.4903259277344, 90.7309455871582)], [(602.4657897949219, 90.3957633972168)], [(603.7971801757812, 90.9417610168457)], [(606.39306640625, 91.54962921142578)], [(608.4457092285156, 91.79508209228516)], [(610.0652770996094, 92.41083908081055)], [(587.3049926757812, 268.5137481689453)], [(587.6962280273438, 270.71117401123047)], [(589.0084838867188, 271.36322021484375)], [(619.8650207519531, 92.30441665649414), (588.7535400390625, 271.3638000488281)], [(621.2707214355469, 92.32811737060547), (589.0105590820312, 271.622314453125)], [(622.6288757324219, 91.44775772094727), (589.3644104003906, 271.6300582885742)], [(623.7484130859375, 91.78638458251953), (588.9454040527344, 271.29334259033203)], [(625.0731506347656, 91.27238082885742), (589.4116821289062, 270.6444854736328)], [(626.4897155761719, 92.41292953491211), (589.6850280761719, 270.60233306884766)], [(627.1671752929688, 92.25128555297852), (589.4307556152344, 270.45365142822266), (695.2348327636719, 74.92931747436523)], [(628.5409545898438, 93.02381134033203), (589.1073303222656, 270.43714141845703), (694.3469543457031, 75.19856262207031)], [(630.7630310058594, 92.91498184204102), (589.6225280761719, 270.6686019897461), (694.9767456054688, 76.05460166931152)], [(632.9879455566406, 93.21758270263672), (590.1309509277344, 271.1461639404297), (694.0518188476562, 75.51299476623535)], [(634.5164184570312, 92.64760208129883), (589.1869506835938, 271.8551559448242), (694.9652709960938, 75.3836727142334), (316.9483184814453, 282.0380554199219)], [(637.1838073730469, 92.28391647338867), (694.9466857910156, 75.51249504089355), (315.9139404296875, 281.8846893310547)], [(640.1426086425781, 92.61783599853516), (694.8787536621094, 73.73772430419922), (314.48760986328125, 282.1305389404297)], [(642.4379272460938, 92.13998794555664), (313.7580108642578, 282.75807189941406), (695.0043334960938, 75.70962524414062)], [(643.5638122558594, 92.12508010864258), (311.73828125, 282.6742706298828)], [(644.8960876464844, 92.45302963256836), (694.3485717773438, 71.42223739624023)], [(647.5674133300781, 92.27509689331055), (974.156494140625, 74.0130443572998), (694.4216003417969, 71.64499855041504)], [(650.99560546875, 92.12645721435547), (694.9258422851562, 70.93244934082031)], [(653.2906494140625, 92.71752548217773), (967.2431945800781, 75.50555038452148), (694.6134948730469, 70.18394660949707), (309.8816833496094, 282.55458068847656)], [(654.55908203125, 93.28221130371094), (965.552978515625, 75.5622329711914), (695.7242126464844, 74.34855270385742), (309.12298583984375, 282.70054626464844)], [(655.5711669921875, 93.55830001831055), (960.6316223144531, 75.93354225158691), (694.6317138671875, 70.16048240661621), (330.5471954345703, 239.20297241210938), (308.11474609375, 282.75927734375)], [(656.7647094726562, 93.25690078735352), (960.1913757324219, 76.18396377563477), (307.4674377441406, 282.8462371826172), (330.2922821044922, 237.56019592285156), (952.9163208007812, 75.91986083984375)], [(657.828369140625, 94.1209831237793), (953.0087280273438, 75.48702430725098), (333.15960693359375, 236.86973571777344), (305.3951873779297, 283.0195770263672), (695.6637573242188, 73.07089805603027)], [(657.8691101074219, 94.55904769897461), (951.4183959960938, 75.01398086547852), (303.95916748046875, 283.15728759765625), (943.6514282226562, 74.41665458679199)], [(658.3125, 94.28607559204102), (948.6936340332031, 75.89575386047363)], [(659.9952697753906, 94.40924453735352), (945.2208251953125, 76.20425415039062), (692.9215087890625, 76.60274505615234)], [(660.6959838867188, 94.17353439331055), (943.359375, 75.1729907989502), (691.5343933105469, 74.8107852935791)], [(662.5606079101562, 94.09958267211914), (941.4485168457031, 76.04992294311523), (690.6531372070312, 75.13665008544922)], [(665.2473449707031, 93.9560432434082), (939.5082702636719, 77.14460182189941)], [(666.7591857910156, 93.53158569335938), (938.3204040527344, 77.12912940979004), (692.2277221679688, 75.9255485534668), (680.9309387207031, 91.65811157226562)], [(669.4932250976562, 92.6383171081543), (935.6232604980469, 77.24880981445312)], [(935.1785888671875, 77.11861991882324), (673.2823791503906, 91.9710693359375), (685.9339904785156, 89.39292526245117)], [(933.1245422363281, 77.83060073852539), (675.3813781738281, 92.19697952270508)], [(676.697998046875, 92.52283096313477), (931.0069580078125, 78.20838356018066)], [(678.686767578125, 92.2300910949707), (927.3473510742188, 78.08165550231934)], [(680.23583984375, 92.26912307739258), (923.0424499511719, 78.65127944946289), (448.18934631347656, 223.24987030029297)], [(684.0106201171875, 92.30947494506836), (921.4100646972656, 79.49243545532227), (451.72552490234375, 224.97279357910156)], [(685.8962707519531, 92.2376480102539), (919.7870788574219, 79.36848831176758)], [(686.8289489746094, 92.25672912597656), (919.3833923339844, 80.05070114135742)], [(687.4752807617188, 92.4818229675293), (919.5245666503906, 80.87651634216309)], [(687.5335998535156, 93.04435348510742), (918.0550231933594, 79.60874366760254)], [(688.8419494628906, 93.0405387878418), (918.3044738769531, 80.51615715026855), (289.1084213256836, 233.4448699951172)], [(691.4876098632812, 92.78393936157227), (918.7453918457031, 82.3192367553711)]]
    max_len = len(max(centers, key=lambda x: len(x)))
    list1 = [[] for _ in range(max_len)]
    for frame in centers:
        for i, cen in enumerate(frame):
            list1[i].append(cen)
        for j in range(len(frame),max_len):
            list1[j].append((None,None))
    list1 = np.array(list1)
    plt.figure()
    for list in list1:
        x_values = list[:,1]
        plt.scatter(range(len(list)), x_values, c='b')
    plt.show()'''


    court_detector = CourtDetector()
    video = cv2.VideoCapture('../videos/vid22.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output videos writer
    out = cv2.VideoWriter(os.path.join('output', 'player2test.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))
    dtype = get_dtype()
    model = DetectionModel(dtype)
    frame_i = 0
    while True:
        ret, frame = video.read()
        frame_i += 1
        if ret:
            if frame_i == 1:
                court_detector.detect(frame)
            court_detector.track_court(frame)

            frame = model.detect_top_persons(frame, court_detector)
            out.write(frame)
            cv2.imshow('df', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        else:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()
    model.find_player_2_box()
