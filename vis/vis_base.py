import torch

from .env import *

__all__ = [
    "__allowed_types__",
    "imT",
    "get_subplots_shape",
]


__allowed_types__ = (np.ndarray, torch.Tensor, list)

imT = TypeVar("imT", np.ndarray, torch.Tensor)  # Image type. Only for function typing


class BaseVis:
    @staticmethod
    def deal_with_img_types(img: imT) -> np.ndarray:
        """Converts image to numpy.ndarray .

        Args:
            img (imT): Image

        Raises:
            TypeError: if the image type is not allowed

        Returns:
            np.ndarray: Image converted to numpy.ndarray
        """
        if not isinstance(img, __allowed_types__):
            raise TypeError(
                f"Expected image of type {__allowed_types__} . Got: {type(img)}"
            )

        if isinstance(img, list):
            warnings.warn(
                "WARNING: Using list of Tensors or Arrays might be slow for big image clips."
            )
            if not all([isinstance(im, (np.ndarray, torch.Tensor)) for im in img]):
                raise TypeError(
                    f"Expected image of type {(np.ndarray, torch.Tensor)} . Got: {[type(im) for im in img]}"
                )
            return np.stack(img, 0)

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, torch.Tensor):
            return img.detach().cpu().numpy()

    @staticmethod
    def read(file_path: Union[str, Path], **kwargs: Any) -> np.ndarray:
        """Loads an image.

        Args:
            file_path (Union[str, Path]): Image file path

        Returns:
            np.ndarray: Image
        """
        raise NotImplementedError

    @staticmethod
    def read_multi(file_paths: List[Union[str, Path]], **kwargs: Any) -> np.ndarray:
        """Loads multiple images.

        Args:
            file_paths (List[Union[str, Path]]): Image file paths

        Returns:
            np.ndarray: Loaded images stacked on dimention 0
        """
        raise NotImplementedError

    @staticmethod
    def write(img: imT, file_path: Union[str, Path], **kwargs: Any) -> None:
        """Saves an image.

        Args:
            img (imT): Image to save
            file_path (Union[str, Path]): Save path
        """
        raise NotImplementedError

    @staticmethod
    def write_multi(
        imgs: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        file_dir: Union[str, Path],
        file_basename: str,
        nlen: int = 4,
        start: int = 0,
        show_bar: bool = False,
        **kwargs,
    ):
        """Saves multiple images.

        Args:
            imgs (Union[imT, List[imT]]): Stacked imgs on the first dimention. imgs[n_imgs,
            ...]
            file_dir (Union[str, Path]): Image save directory
            file_basename (str): Images name before adding '_[images_number]
            nlen (int, optional): Number of zero padding on the image name. Defaults to 4.
        """
        raise NotImplementedError

    @staticmethod
    def show(
        img: imT,
        **kwargs: Any,
    ) -> None:
        """Shows an image.

        Args:
            img (imT): Image
        """
        raise NotImplementedError

    @staticmethod
    def show_multi(
        imgs: imT,
        imgs_shape: str = "BHWC",
        **kwargs: Any,
    ) -> None:
        """Show multiple images.

        Args:
            imgs (imT): Images
            imgs_shape (str, optional): Shape of the passed images. Defaults to "BHWC"  (Batch, Heigth, Width, Channels).


        """
        raise NotImplementedError

    # Other naming
    @classmethod
    def readm(cls, *args, **kwargs) -> np.ndarray:
        "Same as cls.read_multi(...)"
        return cls.read_multi(*args, **kwargs)

    @classmethod
    def writem(cls, *args, **kwargs) -> None:
        "Same as cls.write_multi(...)"
        cls.write_multi(*args, **kwargs)

    @classmethod
    def showm(cls, *args, **kwargs) -> None:
        "Same as cls.show_multi(...)"
        cls.show_multi(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> np.ndarray:
        "Same as cls.read(...)"
        return cls.read(*args, **kwargs)

    @classmethod
    def save(cls, *args, **kwargs) -> None:
        "Same as cls.write(...)"
        cls.write(*args, **kwargs)

    @classmethod
    def display(cls, *args, **kwargs) -> None:
        "Same as cls.show(...)"
        cls.show(*args, **kwargs)

    @classmethod
    def load_multi(cls, *args, **kwargs) -> np.ndarray:
        "Same as cls.read_multi(...)"
        return cls.read_multi(*args, **kwargs)

    @classmethod
    def save_multi(cls, *args, **kwargs) -> None:
        "Same as cls.write_multi(...)"
        cls.write_multi(*args, **kwargs)

    @classmethod
    def display_multi(cls, *args, **kwargs) -> None:
        "Same as cls.show_multi(...)"
        cls.show_multi(*args, **kwargs)

    @classmethod
    def loadm(cls, *args, **kwargs) -> np.ndarray:
        "Same as cls.read_multi(...)"
        return cls.read_multi(*args, **kwargs)

    @classmethod
    def savem(cls, *args, **kwargs) -> None:
        "Same as cls.write_multi(...)"
        cls.write_multi(*args, **kwargs)

    @classmethod
    def displaym(cls, *args, **kwargs) -> None:
        "Same as cls.show_multi(...)"
        cls.show_multi(*args, **kwargs)


def get_subplots_shape(img_no: int) -> Tuple[int, int]:
    """Gets the best combination of lines and columns to plot multiple images.

    Args:
        img_no (int): Number of images

    Returns:
        Tuple[int, int]: Number of lines and columns
    """
    p = np.arange(1, img_no)

    div = img_no / p
    div_int = div[div == div.astype(int)]

    try:
        closest_value = min(
            div_int, key=lambda list_value: abs(list_value - np.sqrt(img_no))
        )
    except ValueError:
        closest_value = 1

    lines = closest_value
    cols = img_no / lines

    return int(lines), int(cols)


# ENDFILE
