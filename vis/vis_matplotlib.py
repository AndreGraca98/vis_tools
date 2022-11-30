from einops import rearrange
from matplotlib.figure import Figure

from .env import *
from .vis_base import BaseVis, get_subplots_shape, imT

__all__ = ["PLT"]


class PLT(BaseVis):
    @staticmethod
    def read(file_path: Union[str, Path], **kwargs) -> np.ndarray:
        """Loads an image.

        Note:
            PNG file is loaded as type float even if it is saved as other types.
            ex: saved_png_img(uint8), max=255 -> PLT.read(saved_png_img) ->
            loaded_img(float32), max=1.0

        Args:
            file_path (Union[str, Path]): Image file path

        Returns:
            np.ndarray: Image
        """
        return plt.imread(str(file_path), **kwargs)

    @staticmethod
    def read_multi(
        file_paths: List[Union[str, Path]], verbose: bool = False, **kwargs
    ) -> np.ndarray:
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
            imgs.append(PLT.read(file_path=file_path, **kwargs))
        return np.stack(imgs, 0)  # type: ignore

    @staticmethod
    def write(
        img: imT,
        file_path: Union[str, Path],
        plt_kwargs: Dict[str, Any] = dict(),
        **save_kwargs: Any,
    ) -> None:
        """Saves an image.

        Args:
            img (imT): Image to save
            file_path (Union[str, Path]): Save path
            plt_kwargs (Dict[str, Any], optional): PLT.show(...) kwargs. Defaults to dict().
        """

        img = PLT.deal_with_img_types(img)  # type: ignore

        # If file directory does not exist create it and its parent dirs
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)

        fig = PLT.show(img, **plt_kwargs)
        fig.savefig(str(file_path), **save_kwargs)

    @staticmethod
    def write_multi(
        imgs: Union[imT, List[imT]],
        file_dir: Union[str, Path],
        file_basename: str,
        nlen: int = 4,
        start: int = 0,
        show_bar: bool = False,
        plt_kwargs: Dict[str, Any] = dict(),
        **save_kwargs: Any,
    ):
        """Saves multiple images.

        Args:
            imgs (Union[imT, List[imT]]): Stacked imgs on the first dimention. imgs[n_imgs,
            ...]
            file_dir (Union[str, Path]): Image save directory
            file_basename (str): Images name before adding '_[images_number].png'
            nlen (int, optional): Number of zero padding on the image name. Defaults to 4.
            start (int, optional): Image name number to start counting . Defaults to 0.
            show_bar (bool, optional): Show progress bar. Defaults to False.
            plt_kwargs (Dict[str, Any], optional): PLT.show(...) kwargs. Defaults to dict().
        """
        for n, img in enumerate(
            tqdm(imgs, desc=f"Saving images to {file_dir}", disable=not show_bar),
            start=start,
        ):
            PLT.write(
                img=img,
                file_path=Path(file_dir)
                / (str(file_basename) + "_{:0{nlen}}.png".format(n, nlen=int(nlen))),
                plt_kwargs=plt_kwargs,
                **save_kwargs,
            )

    @staticmethod
    def show(
        img: imT,
        cmap: str = "jet",
        figsize: Union[int, float, str, List, Tuple] = 10,
        title="",
        method: str = "imshow",
        **kwargs,
    ) -> Figure:
        """Shows an image.

        Args:
            img (np.ndarray): Image
            imgname (str, optional): Image name. Defaults to "Image".
            wait_time (int, optional): If None, does not wait. If 0, waits for
            user input. Else waits for the amount of wait_time (miliseconds). Defaults to None.


        Args:
            img (imT): _description_
            cmap (str, optional): _description_. Defaults to "jet".
            figsize (Union[int, float, str, List, Tuple], optional): _description_. Defaults to 10.
            title (str, optional): _description_. Defaults to "".
            method (str, optional): _description_. Defaults to "imshow".

        Raises:
            TypeError: _description_

        Returns:
            Figure: _description_
        """
        img = PLT.deal_with_img_types(img)  # type: ignore

        if img.ndim > 2:
            assert img.shape[~0] in [
                1,
                3,
                4,
            ], f"Expected number of channels to be one of [1,3,4]. Got: channels={img.shape[~0]}"

        if isinstance(figsize, (int, float, str)):
            figsize = (int(figsize), int(figsize))
        elif isinstance(figsize, (tuple, list)):
            ...
        else:
            raise TypeError(
                f"Expected figsize to be of type: int, float, str, tuple or list. Got: {type(figsize)} ."
            )

        fig = plt.figure(figsize=figsize)

        show_methods = dict(imshow=plt.imshow, matshow=plt.matshow)
        show_methods[method](img, cmap=cmap, **kwargs)

        plt.title(str(title))

        return fig

    @staticmethod
    def show_multi(
        imgs: imT,
        imgs_shape: str = "BHWC",
        **kwargs,
    ) -> Figure:
        """View frames with matplotlib. Supports 3,4 and 5 dimentional
        image arrays. If array with five dimentions is passed
        perform the mean over the N dimention.

        Uagage:
            imgs_shape='BCNHW'
            imgs = np.random.random((32,3,12,224,224))
            PLT.show_multi(imgs, imgs_shape)

        Args:
            imgs (imT): Images
            imgs_shape (str, optional): Shape of the passed images . Defaults to "BHWC"  (Batch, Heigth, Width, Channels).
        """
        imgs = PLT.deal_with_img_types(imgs)  # type:ignore
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
                return PLT.show(imgs, **kwargs)
            imgs = np.expand_dims(imgs, ~0)  # Add channel dim for gray img

        # Create image grid
        img_no = imgs.shape[0]
        lines, cols = get_subplots_shape(img_no)

        himgs = [
            np.hstack(imgs[line * cols : line * cols + cols, ...])  # type:ignore .
            for line in range(lines)
        ]
        vimgs = np.vstack(himgs)

        return PLT.show(vimgs, **kwargs)


# ENDFILE
