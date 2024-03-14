import sys
sys.path.append('/content/TennisProject/src')

import numpy as np
import pandas as pd
import cv2
import torch
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import argrelextrema


from ball_tracker_net import BallTrackerNet
from detection import center_of_box
from utils import get_video_properties


def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Combine three frames into one input tensor for detecting the ball
    """

    # Resize and type converting for each frame
    img = cv2.resize(frame1, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # resize it
    img1 = cv2.resize(frame2, (width, height))
    # input must be float type
    img1 = img1.astype(np.float32)

    # resize it
    img2 = cv2.resize(frame3, (width, height))
    # input must be float type
    img2 = img2.astype(np.float32)

    # combine three imgs to  (width , height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)

##-----------------------------------------------xav-----------------------------------
def from_2d_array_to_nested(
    X, index=None, columns=None, time_index=None, cells_as_numpy=False
):
    """Convert 2D dataframe to nested dataframe.

    Convert tabular pandas DataFrame with only primitives in cells into
    nested pandas DataFrame with a single column.

    Parameters
    ----------
    X : pd.DataFrame

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series

    index : array-like, shape=[n_samples], optional (default = None)
        Sample (row) index of transformed DataFrame

    time_index : array-like, shape=[n_obs], optional (default = None)
        Time series index of transformed DataFrame

    Returns
    -------
    Xt : pd.DataFrame
        Transformed DataFrame in nested format
    """
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series

    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def find_bounce():
    data = pd.read_csv("test_df.csv", usecols=["y"])
    y_values = np.array(data[:])
    print("------------------------")
    #print(y_values)

    ##-------------------lissage ball coordinate-------------------------
    y_values = y_values.flatten()
    series = pd.Series(y_values)
    print("-----------------------------")
    #print(series)
    print("------------------------------------")

    # Apply exponential smoothing with smoothing factor alpha
    alpha = 0.2  # Adjust the value of alpha as needed
    smoothed_values = series.ewm(alpha=alpha).mean()

    # Convert the smoothed values back to a numpy array
    y_values = smoothed_values.values
    #print(y_values)
    ##----------------------fin lissage--------------------

    y = np.array(y_values, dtype=float)

    # Initialize an array of NaNs for the second derivative
    first_derivative = np.full(y.shape, np.nan)
    second_derivative = np.full(y.shape, np.nan)

    # Compute the first and second derivatives using central difference, skipping NaN values
    #delta_t = (1/30)
    for i in range(1, len(y)):
        if not np.isnan(y[i-1]) and not np.isnan(y[i]):
            first_derivative[i] = (y[i] - y[i-1])

    for i in range(1, len(first_derivative)):
            if not np.isnan(first_derivative[i-1]) and not np.isnan(first_derivative[i]):
                second_derivative[i] = (first_derivative[i] - first_derivative[i-1]) / 0.2


  #      for i in range(len(first_derivative)):
  #          print(i, first_derivative[i])

   #     for i in range(len(second_derivative)):
   #         print(i, second_derivative[i])


    plt.xlabel('Frame Index')
    plt.ylabel('Y-Index Position')
    plt.title('Ball and Players Y-Index Positions Over Frames')
    plt.legend()
    plt.savefig("positions_over_frames.jpg")

    plt.figure()
    plt.scatter(range(len(y_values)), [((y/10)-10) if y is not None else 0 for y in y_values], marker='o', label='Ball', color='blue')
    plt.plot(range(len(y_values)), first_derivative, label='1st deriv', color='r')
    plt.plot(range(len(y_values)), second_derivative, label='2nd deriv', color='g')
    plt.xlabel('Frame Index')
    plt.ylabel('derivatives')
    plt.title('ball, first derivative and second derivative over frame')
    plt.legend()
    plt.savefig("derivative_over_frames.jpg")


    ##-------------------------lissage derive premiere---------------------------------------------
    first_derivative = first_derivative.flatten()
    series = pd.Series(first_derivative)
    #print("-----------------------------")
    #print(series)
    #print("------------------------------------")

    # Apply exponential smoothing with smoothing factor alpha
    alpha = 0.2  # Adjust the value of alpha as needed
    smoothed_values = series.ewm(alpha=alpha).mean()

    # Convert the smoothed values back to a numpy array
    first_derivative = smoothed_values.values
    #print(y_values)
    ##---------------------------fin lissage----------------------------------------

    #for i in range(1, len(first_derivative) - 1):
    #    if not np.isnan(first_derivative[i-1]) and not np.isnan(first_derivative[i]) and not np.isnan(first_derivative[i+1]):
    #        second_derivative[i] = (first_derivative[i+1] - 2*first_derivative[i] + first_derivative[i-1])

    ##-----------------------maximums locaux---------------------------
    y_values = np.array(first_derivative)

    # Trouver les indices des maximums locaux
    local_max_indices = argrelextrema(y_values, np.greater)

    # Les maximums locaux correspondants aux indices trouvés
    local_max_values = y_values[local_max_indices]

    print("Indices des maximums locaux :", local_max_indices)
    print("Valeurs des maximums locaux :", local_max_values)

    return local_max_indices
    ##------------------------fin max locaux---------------------------

##------------------------------------------fin xav--------------------------------------------------


class BallDetector:
    """
    Ball Detector model responsible for receiving the frames and detecting the ball
    """
    def __init__(self, model_saved_state, out_channels=2):
        # Construct absolute path to the weights file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_directory, '..', 'saved states', model_saved_state)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load TrackNet model weights
        self.detector = BallTrackerNet(out_channels=out_channels)
        saved_state_dict = torch.load(weights_path)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):
        """
        After receiving 3 consecutive frames, the ball will be detected using TrackNet model
        :param frame: current frame
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # detect only in 3 frames were given
        if self.last_frame is not None:
            # combine the frames into 1 input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            # Inference (forward pass)
            x, y = self.detector.inference(frames)
            if x is not None:
                # Rescale the indices to fit frame dimensions
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)

                # Check distance from previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    def mark_positions(self, frame, mark_num=4, frame_num=None, ball_color='yellow'):
        """
        Mark the last 'mark_num' positions of the ball in the frame
        :param frame: the frame we mark the positions in
        :param mark_num: number of previous detection to mark
        :param frame_num: current frame number
        :param ball_color: color of the marks
        :return: the frame with the ball annotations
        """
        bounce_i = None
        # if frame number is not given, use the last positions found
        if frame_num is not None:
            q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
            for i in range(frame_num - mark_num + 1, frame_num + 1):
                if i in self.bounces_indices:
                    bounce_i = i - frame_num + mark_num - 1
                    break
        else:
            q = self.xy_coordinates[-mark_num:, :]
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        # Mark each position by a circle
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(pil_image)
                if bounce_i is not None and i == bounce_i:
                    draw.ellipse(bbox, outline='red')
                else:
                    draw.ellipse(bbox, outline=ball_color)

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def calculate_ball_positions(self):
        return self.xy_coordinates

    #---------------------------------------------------------xav--------------------------------------------------
    def calculate_ball_position_top_view(self, court_detector):
        inv_mats = court_detector.game_warp_matrix
        xy_coordinates_top_view = []
        for i, pos in enumerate(self.xy_coordinates):
            if pos[0]==None:
                ball_pos = np.array([100.25, 100.89]).reshape((1, 1, 2))
            else:
                ball_pos = np.array([pos[0], pos[1]]).reshape((1, 1, 2))
 #           print(ball_pos)     ## -------------------------xav-----------------------------
            ball_court_pos = cv2.perspectiveTransform(ball_pos, inv_mats[i]).reshape(-1)
            xy_coordinates_top_view.append(ball_court_pos)
        return xy_coordinates_top_view
    #---------------------------------------------------------fin xav----------------------------------------------
    #---------------------------------------------------------xav--------------------------------------------------
    def diff_xy(self, coords):
        coords = coords.copy()
        diff_list = []
        for i in range(0, len(coords)-1):
            if coords[i] is not None and coords[i+1] is not None:
                point1 = coords[i]
                point2 = coords[i+1]
                diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
                diff_list.append(diff)
            else:
                diff_list.append([None, None])

        xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array([x[1] if x is not None else np.nan for x in diff_list])

        return xx, yy

    def remove_outliers(self, x, y, coords):
        # Filtrer les valeurs None de x et y
        x_filtered = [val for val in x if val is not None]
        y_filtered = [val for val in y if val is not None]

        # Vérifier si x_filtered et y_filtered ne sont pas vides
        if x_filtered and y_filtered:
            # Comparer avec 50
            ids = set(np.where(np.array(x_filtered) > 50)[0]) & set(np.where(np.array(y_filtered) > 50)[0])

            # Rétablir les indices dans la liste complète coords
            for id in ids:
                # L'indice dans x_filtered et y_filtered
                id_global = x.index(x_filtered[id])

                # Récupérer les valeurs des voisins
                left = coords[id_global-1] if id_global > 0 else None
                middle = coords[id_global]
                right = coords[id_global+1] if id_global < len(coords)-1 else None

                # Traitement des valeurs None
                if left is None:
                    left = [0]
                if right is None:
                    right = [0]
                if middle is None:
                    middle = [0]

                # Trouver la valeur maximale entre left, middle et right
                MAX = max(map(list, (left, middle, right)))

                # Mettre à None la coordonnée maximale dans coords
                if MAX != [0]:
                    try:
                        coords[coords.index(tuple(MAX))] = None
                    except ValueError:
                        coords[coords.index(MAX)] = None


    def interpolation(self, coords):
        coords =coords.copy()
        x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

        xxx = np.array(x) # x coords
        yyy = np.array(y) # y coords

        nons, yy = nan_helper(xxx)
        xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
        nans, xx = nan_helper(yyy)
        yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

        newCoords = [*zip(xxx,yyy)]

        return newCoords



    #---------------------------------------------------------fin xav----------------------------------------------

    def show_y_graph(self, player_1_boxes, player_2_boxes):

        player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
        player_1_y_values = player_1_centers[:, 1] - np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])

        player_2_centers = np.array([center_of_box(box) if box[0] is not None else [None, None] for box in player_2_boxes])
        player_2_y_values = player_2_centers[:, 1]

        y_values = self.xy_coordinates[:, 1].copy()
        x_values = self.xy_coordinates[:, 0].copy()

        plt.figure()
        plt.scatter(range(len(y_values)), y_values, marker='o', label='Ball', color='blue')
        plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r', marker='o', linestyle='-', label='Player 1')
        plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g', marker='o', linestyle='-', label='Player 2')









if __name__ == "__main__":
    ball_detector = BallDetector('saved states/tracknet_weights_lr_1.0_epochs_150_last_trained.pth')
    cap = cv2.VideoCapture('../videos/vid1.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)


    cap.release()
    cv2.destroyAllWindows()

    from scipy.interpolate import interp1d

    y_values = ball_detector.xy_coordinates[:,1]

    new = signal.savgol_filter(y_values, 3, 2)

    x = np.arange(0, len(new))
    indices = [i for i, val in enumerate(new) if np.isnan(val)]
    x = np.delete(x, indices)
    y = np.delete(new, indices)
    f = interp1d(x, y, fill_value="extrapolate")
    f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(y_values), num=len(y_values), endpoint=True)
    plt.plot(np.arange(0, len(new)), new, 'o',xnew,
             f2(xnew), '-r')
    plt.legend(['data', 'inter'], loc='best')
    plt.show()

    positions = f2(xnew)
    peaks, _ = find_peaks(positions, distance=30)
    a = np.diff(peaks)
    plt.plot(positions)
    plt.plot(peaks, positions[peaks], "x")
    plt.show()
