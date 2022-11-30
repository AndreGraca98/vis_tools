from einops import rearrange

from .env import *
from .vis_base import BaseVis, get_subplots_shape, imT

__all__ = ["cv2_Colormaps", "CV2"]

cv2_Colormaps = EDict(
    AUTUMN=cv2.COLORMAP_AUTUMN,
    BONE=cv2.COLORMAP_BONE,
    JET=cv2.COLORMAP_JET,
    WINTER=cv2.COLORMAP_WINTER,
    RAINBOW=cv2.COLORMAP_RAINBOW,
    OCEAN=cv2.COLORMAP_OCEAN,
    SUMMER=cv2.COLORMAP_SUMMER,
    SPRING=cv2.COLORMAP_SPRING,
    COOL=cv2.COLORMAP_COOL,
    HSV=cv2.COLORMAP_HSV,
    PINK=cv2.COLORMAP_PINK,
    HOT=cv2.COLORMAP_HOT,
    PARULA=cv2.COLORMAP_PARULA,
    MAGMA=cv2.COLORMAP_MAGMA,
    INFERNO=cv2.COLORMAP_INFERNO,
    PLASMA=cv2.COLORMAP_PLASMA,
    VIRIDIS=cv2.COLORMAP_VIRIDIS,
    CIVIDIS=cv2.COLORMAP_CIVIDIS,
    TWILIGHT=cv2.COLORMAP_TWILIGHT,
    TWILIGHT_SHIFTED=cv2.COLORMAP_TWILIGHT_SHIFTED,
    TURBO=cv2.COLORMAP_TURBO,
    DEEPGREEN=cv2.COLORMAP_DEEPGREEN,
)


class CV2(BaseVis):
    @staticmethod
    def read(file_path: Union[str, Path], *cv2_imread_args: Any) -> np.ndarray:
        """Loads an image.

        Args:
            file_path (Union[str, Path]): Image file path

        Returns:
            np.ndarray: Image
        """
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED, *cv2_imread_args)
        if img.ndim == 3 and img.shape[~0] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread assumes BGR
        return img

    @staticmethod
    def read_multi(
        file_paths: List[Union[str, Path]], *cv2_imread_args: Any, verbose: bool = False
    ) -> List[np.ndarray]:
        """Load multiple images.

        Args:
            file_paths (List[Union[str, Path]]): Image file paths
            verbose (bool, optional): Print image path while loading. Defaults to False.

        Returns:
            np.ndarray: Images
        """
        imgs = []
        for file_path in file_paths:
            if verbose:
                print(f"Reading {file_path}")
            imgs.append(CV2.read(file_path=file_path, *cv2_imread_args))
        return np.stack(imgs, 0)  # type: ignore

    @staticmethod
    def write(img: imT, file_path: Union[str, Path], *cv2_imwrite_args: Any) -> None:
        """Saves an image.

        Args:
            img (imT): Image to save
            file_path (Union[str, Path]): Save path
        """
        img = CV2.deal_with_img_types(img)  # type: ignore
        # If file directory does not exist create it and its parent dirs
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)

        if img.ndim == 3 and img.shape[~0] == 3:
            img = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB  # type:ignore
            )  # cv2.imread assumes BGR

        cv2.imwrite(str(file_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0, *cv2_imwrite_args])  # type: ignore

    @staticmethod
    def write_multi(
        imgs: Union[imT, List[imT]],
        file_dir: Union[str, Path],
        file_basename: str,
        *cv2_imwrite_args: Any,
        nlen: int = 4,
        start: int = 0,
        show_bar: bool = False,
    ) -> None:
        """Saves multiple images.

        Args:
            imgs (Union[imT, List[imT]]): Stacked imgs on the first dimention. imgs[n_imgs,...]
            file_dir (Union[str, Path]): Image save directory
            file_basename (str): Images name before adding '_[images_number].png'
            nlen (int, optional): Number of zero padding on the image name. Defaults to 4.
            start (int, optional): Image name number to start counting . Defaults to 0.
            show_bar (bool, optional): Show progress bar. Defaults to False.

        """
        for n, img in enumerate(
            tqdm(imgs, desc=f"Saving images to {file_dir}", disable=not show_bar),
            start=start,
        ):
            CV2.write(
                img=img,
                file_path=Path(file_dir)
                / (str(file_basename) + "_{:0{nlen}}.png".format(n, nlen=int(nlen))),
                *cv2_imwrite_args,
            )

    @staticmethod
    def show(
        img: imT,
        resized_shape: Union[Tuple[int, int], None] = None,
        wait_time: int = 1,
        title: str = "Image",
    ) -> None:
        """Shows the image for a specified time

        Args:
            img (np.ndarray): Image
            title (str, optional): Image name. Defaults to "Image".
            wait_time (int, optional): If None, does not wait. If 0, waits for
            user input. Else waits for the amount of wait_time (miliseconds). Defaults to None.


        Args:
            img (imT): Image
            title (str, optional): Image name. Defaults to "Image".
            resized_shape (Union[Tuple[int, int], None], optional): Resized image shape. If None shows the full image. Defaults to None.
            wait_time (int, optional): Time to wait in milliseconds. If 0 waits for user input. Defaults to 0.
        """
        img = CV2.deal_with_img_types(img)  # type:ignore

        if img.ndim > 2:
            assert img.shape[~0] in [
                1,
                3,
                4,
            ], f"Expected number of channels to be one of [1,3,4]. Got: channels={img.shape[~0]}"

        if resized_shape is not None:
            img = cv2.resize(img, dsize=resized_shape)  # type:ignore

        if img.ndim == 3 and img.shape[~0] == 3:
            img = cv2.cvtColor(
                img, cv2.COLOR_RGB2BGR  # type:ignore . cv2.imread assumes BGR
            )

        cv2.imshow(title, img)

        cv2.waitKeyEx(wait_time)
        cv2.destroyAllWindows()

    @staticmethod
    def show_multi(imgs: np.ndarray, imgs_shape: str = "BHWC", **kwargs) -> None:
        """View frames with opencv. If array with five dimentions is passed
        perform the mean over the N dimention.

        Uagage:
            imgs_shape='BCNHW'
            imgs = np.random.random((32,3,12,224,224))
            CV2.show_multi(imgs, imgs_shape, wait_time=0)

        Args:
            imgs (np.ndarray): Image array
            imgs_shape (str, optional): Shape of the passed image array. Defaults to
            "BCNHW" (Batch, Channels, Number of samples, Heigth, Width).
            wait_time (int, optional): If None, does not wait. If 0, waits for
            user input. Else waits for the amount of wait_time (miliseconds). Defaults to None.

        """
        imgs = CV2.deal_with_img_types(imgs)  # type:ignore
        imgs_shape = imgs_shape.upper()

        assert len(imgs_shape) == imgs.ndim and imgs.ndim in [
            3,
            4,
            5,
        ], f"Expected imgs_shape and imgs.ndim of length 3, 4 or 5. Got: len(imgs_shape)={len(imgs_shape)} ; imgs.ndim={imgs.ndim}"

        assert all(
            [c in "BCNHW" for c in imgs_shape]
        ), f'Expected all letters to be in {list("BCNHW")}. Got: {imgs_shape}'

        assert len(imgs_shape) == len(
            set(imgs_shape)
        ), f"Expected all letters to be different. Got: {imgs_shape}"

        assert (
            "H" in imgs_shape.upper() and "W" in imgs_shape.upper()
        ), f"Expected height(H) and width(W) to be on imgs_shape. Got: {imgs_shape}"

        if imgs.ndim == 5:
            new_order = "bnhwc"
        elif imgs.ndim == 4:
            new_order = "bhwc"
        elif imgs.ndim == 3:
            if "B" in imgs_shape.upper():
                new_order = "bhw"  # batch of grayscale images
            else:
                new_order = "hwc"  # single rgb image

        imgs = rearrange(
            imgs,
            f'{" ".join(imgs_shape.upper())} -> {" ".join(new_order.upper())}',  # type: ignore
        )

        if imgs.ndim == 5:
            imgs = imgs.mean(1)  # type:ignore . mean over n dim
        elif imgs.ndim == 3:
            if "C" in imgs_shape.upper():  # single rgb image
                return CV2.show(imgs, **kwargs)
            imgs = np.expand_dims(imgs, ~0)  # Add channel dim for gray img

        # Create image grid
        img_no = imgs.shape[0]
        lines, cols = get_subplots_shape(img_no)

        himgs = [
            np.hstack(imgs[line * cols : line * cols + cols, ...])  # type:ignore .
            for line in range(lines)
        ]
        vimgs = np.vstack(himgs)

        return CV2.show(vimgs, **kwargs)


# ENDFILE
