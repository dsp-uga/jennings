def extract_features(vid, feature_fns):
    '''Extract features from an input video.

    Args:
        vid:
            A numpy array of shape (frames, height, width).
        feature_fns:
            A list of functions to extract features from the video.
            Each must output a feature map of shape (height, width).

    Returns:
        A numpy array of shape (height, width, features).
    '''
    features = [fn(vid) for fn in feature_fns]
    return np.stack(features, -1)


def first_frame(vid):
    '''Extracts the first frame of the video.

    Args:
        vid: A numpy array of shape (frames, height, width).

    Returns:
        The first frame of `vid`.

    See:
        `jennings.features.extract_features`
    '''
    return vid[0]
