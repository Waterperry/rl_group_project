__author__ = 'tb791'

import numpy as np
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor


# function to get the Euclidean distance from the car to the grass
def distance_preprocess(state: np.ndarray):
    assert state.shape == (96, 96, 3), "Invalid input passed to distance preprocessor."
    car_centre = (71, 48)
    np_car_centre = np.array(car_centre)
    # (-1, 0) is 'forward', (0, 1) is right, (0, -1) is left...
    deltas = np.array(((-1, 0), (0, 1), (0, -1), (-1, 2), (-1, -2)))
    rays = []
    for d_idx, delta in enumerate(deltas):
        curr = car_centre

        while state[curr][1] < 200:
            if not (0 < curr[0] + delta[0] < 96 and 0 < curr[1] + delta[1] < 96):
                break
            curr = curr[0] + delta[0], curr[1] + delta[1]

        curr = np.array(curr)
        distance = np.linalg.norm(curr - np_car_centre)
        rays.append(distance)

    return np.array(rays)


def tf_distance_preprocess2(state: Tensor):
    state = state[0]
    assert state.shape == (96, 96, 3), "Invalid input passed to distance preprocessor."
    car_centre = (71, 48)
    np_car_centre = convert_to_tensor(car_centre)
    # (-1, 0) is 'forward', (0, 1) is right, (0, -1) is left...
    deltas = convert_to_tensor(((-1, 0), (0, 1), (0, -1), (-1, 2), (-1, -2)))
    rays = tf.zeros(deltas.shape[0])
    for d_idx, delta in enumerate(deltas):
        curr = car_centre

        while state[curr][1] < 200:
            if not (0 < curr[0] + delta[0] < 96 and 0 < curr[1] + delta[1] < 96):
                break
            curr = curr[0] + delta[0], curr[1] + delta[1]

        curr = convert_to_tensor(curr)
        distance = tf.linalg.norm(curr - np_car_centre)
        rays[d_idx] = distance

    return tf.expand_dims(rays, axis=0)



car_centre = (71, 48)
# go forward 2 pixels and left 1.
forward_left_ys = list(range(car_centre[0]-2, 0, -2))
forward_left_xs = list(range(car_centre[1]-1, 0, -1))[:len(forward_left_ys)]
assert len(forward_left_ys) == len(forward_left_xs)
forward_left_coords = convert_to_tensor(list(zip(forward_left_ys, forward_left_xs)))

# go forward 2 pixels and right 1
forward_right_ys = list(range(car_centre[0]-2, 0, -2))
forward_right_xs = list(range(car_centre[1]+1, 96, 1))[:len(forward_right_ys)]
assert len(forward_right_ys) == len(forward_right_xs)
forward_right_coords = convert_to_tensor(list(zip(forward_right_ys, forward_right_xs)))


def tf_distance_preprocess(state: Tensor):
    # we assume mismatch = batched input. if not, we need more complex checks here...
    if state.shape != (96, 96, 3):
        state = state[0]
    assert state.shape == (96, 96, 3), "Invalid input to distance preprocessor"

    greens = state[..., 1]

    slice_left = greens[71, :48] >= 200
    slice_right = greens[71, 48:] >= 200
    slice_forward = greens[:71, 48] >= 200
    slice_diag_left = tf.gather_nd(greens, forward_left_coords) >= 200
    slice_diag_right = tf.gather_nd(greens, forward_right_coords) >= 200

    first_green_right = tf.argmax(slice_right)
    first_green_left = tf.argmax(tf.reverse(slice_left, axis=(0, )))
    first_green_forward = tf.argmax(tf.reverse(slice_forward, axis=(0, )))
    first_green_diag_left = tf.argmax(slice_diag_left)
    first_green_diag_right = tf.argmax(slice_diag_right)

    first_green_forward = tf.cond(tf.reduce_max(first_green_forward) == tf.constant(0, dtype=tf.int64),
                                  lambda: tf.convert_to_tensor(71, dtype=tf.int64), lambda: first_green_forward)

    first_green_left = tf.cond(tf.reduce_max(first_green_left) == tf.constant(0, dtype=tf.int64),
                               lambda: tf.convert_to_tensor(48, dtype=tf.int64), lambda: first_green_left)

    first_green_right = tf.cond(tf.reduce_max(first_green_right) == tf.constant(0, dtype=tf.int64),
                                lambda: tf.convert_to_tensor(48, dtype=tf.int64), lambda: first_green_right)

    output = tf.stack((first_green_left, first_green_right, first_green_forward,
                       first_green_diag_left, first_green_diag_right))

    return tf.expand_dims(output, axis=0)
