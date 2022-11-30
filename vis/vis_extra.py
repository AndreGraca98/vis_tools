from .env import *
from .vis_opencv import CV2, cv2_Colormaps

__all__ = [
    "create_clip_from_dir",
    "get_frames_from_video",
    "extract_frames_from_video",
    "extract_frames_from_videos_dir",
    "get_pad_size",
    "generate_gradient_2d",
    "generate_circle",
]


def create_clip_from_dir(
    _dir: str,
    out_path: str,
    frame_duration: Union[float, None] = None,
    clip_duration: Union[float, None] = None,
    add_frame_no: bool = True,
    cmap: str = "turbo",
    show_bar: bool = True,
) -> None:
    """Creates a video or gif from the images in a directory

    Args:
        _dir (str): Directory with the images
        out_path (str): Out file path
        frame_duration (Union[float, None], optional): Frame duration (seconds). Defaults to None.
        clip_duration (Union[float, None], optional): Clip duration (seconds). Defaults to None.
        add_frame_no (bool, optional): Add numbers on the bottom indicating the frame number. Defaults to True.
        cmap (str, optional): Colormap to save the video. Defaults to "turbo".
        show_bar (bool, optional): Show progress bar. Defaults to True.
    """
    assert (frame_duration is None and clip_duration is not None) or (
        frame_duration is not None and clip_duration is None
    ), f"Expected only one of the following: frame_duration={frame_duration} , clip_duration={clip_duration}"

    filenames = sorted(Path(_dir).rglob("*.*"))
    frames = CV2.read_multi(filenames)  # type:ignore

    dur = max(
        0.01, frame_duration or clip_duration / len(frames)  # type: ignore
    )  # Limit to 100fps
    print(f"Set frame duration to {dur:.3f} s. Total GIF time is {dur*len(frames)} s")

    with imageio.get_writer(f"{out_path}", mode="I", fps=1 / dur) as writer:
        for i, frame in tqdm(
            enumerate(frames),
            desc=f"Creating video file with {len(frames)} frames... ",
            disable=not show_bar,
        ):
            frame = cv2.applyColorMap(frame, colormap=cv2_Colormaps[cmap.upper()])
            if add_frame_no:
                frame = cv2.putText(
                    frame,
                    f"{i:03}",
                    org=(
                        int(frame.shape[0] - frame.shape[0] * 0.075),
                        int(frame.shape[1] - frame.shape[1] * 0.0125),
                    ),
                    color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    thickness=1,
                )
            writer.append_data(frame)


################################################################################


def get_frames_from_video(path: Union[str, Path]) -> List[np.ndarray]:
    """Extract frames from a video

    Args:
        path (Union[str, Path]): Video path

    Returns:
        List[np.ndarray]: Frames
    """
    assert Path(path).is_file(), f"File {path} does not exist"
    capture = cv2.VideoCapture(path)

    frames = []
    success, frame = capture.read()
    while success:
        frames.append(frame)
        success, frame = capture.read()

    return frames


def extract_frames_from_video(
    video_path: Union[str, Path], save_path: Union[str, Path], show_bar: bool = False
) -> None:
    """Extract frames from a video and save them

    Args:
        video_path (Union[str, Path]): Video path
        save_path (Union[str, Path]): Images save path
        show_bar (bool, optional): Show progress bar. Defaults to False.
    """
    frames = get_frames_from_video(path=video_path)

    Path(save_path).mkdir(exist_ok=True, parents=True)

    CV2.write_multi(
        imgs=np.stack(frames, 0),
        file_dir=save_path,
        file_basename="image",
        nlen=3,
        show_bar=show_bar,
    )


def extract_frames_from_videos_dir(
    videos_dir: Union[Path, str],
    images_dir: Union[Path, str],
    pattern: str = "*.avi",
    show_bar: bool = False,
) -> None:
    """Extract frames from a directory with videos and save them

    Args:
        videos_dir (Union[Path, str]): Videos directory
        images_dir (Union[Path, str]): Images save directory
        pattern (str, optional): Video names pattern. Defaults to "*.avi".
        show_bar (bool, optional): Show progress bar. Defaults to False.
    """
    video_paths = Path(videos_dir).glob(pattern)

    for path in tqdm(video_paths, ncols=80):
        save_path = Path(images_dir) / path.stem

        extract_frames_from_video(
            video_path=path, save_path=save_path, show_bar=show_bar
        )


################################################################################


def get_pad_size(in_shape=600, out_shape=224):
    count = 0
    while True:
        pad = 0.5 * ((out_shape * count) - in_shape)
        if pad >= 0:
            break
        count += 1

    return int(pad)


################################################################################


def generate_gradient_2d(shape=(224, 224), hi=1.0, wi=1.0) -> np.ndarray:
    """Creates a 2D gradient array

    Args:
        shape (tuple, optional): Gradient shape. Defaults to (224,224).
        hi (int, optional): Height intensity. Defaults to 1.
        wi (int, optional): Width intensity. Defaults to 1.

    Returns:
        np.ndarray: 2D Gradient array
    """
    i, j = np.indices(shape)
    gradient = hi * i + wi * j
    return gradient / gradient.max()


def generate_circle(shape=(200, 200), center=(100, 100), r=0, R=0):
    xx, yy = np.mgrid[: shape[0], : shape[1]]
    # circles contains the squared distance to the (100, 100) point
    # we are just using the circle equation learnt at school
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape

    donut = np.logical_and(circle <= (shape[0] + r), circle > (shape[1] - R))

    return donut.astype(float)


################################################################################

# ENDFILE
