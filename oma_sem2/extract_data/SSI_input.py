import numpy as np

def build_input_matrix(extracted):
    data_list = []

    for sensor, axes in extracted.items():
        for axis in ["X", "Y"]:
            if axes[axis] is not None:
                data_list.append(axes[axis])

    min_len = min(len(arr) for arr in data_list)

    trimmed = [arr[:min_len] for arr in data_list]

    return np.column_stack(trimmed)
