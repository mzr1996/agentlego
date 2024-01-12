from typing import Sequence, Tuple, Union

from agentlego.schema import Annotated, Info
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class OCR(BaseTool):
    """A tool to recognize the optical characters on an image.

    Args:
        lang (str | Sequence[str]): The language to be recognized.
            Defaults to 'en'.
        device (str | bool): The device to load the model. Defaults to True,
            which means automatically select device.
        **read_args: Other keyword arguments for read text. Please check the
            `EasyOCR docs <https://www.jaided.ai/easyocr/documentation/>`_.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = 'This tool can recognize all text on the input image.'

    @require('easyocr')
    def __init__(self,
                 lang: Union[str, Sequence[str]] = 'en',
                 device: Union[bool, str] = True,
                 toolmeta=None,
                 **read_args):
        super().__init__(toolmeta=toolmeta)
        if isinstance(lang, str):
            lang = [lang]
        self.lang = list(lang)
        read_args.setdefault('decoder', 'beamsearch')
        read_args.setdefault('paragraph', True)
        self.read_args = read_args
        self.device = device

    def setup(self):
        import easyocr
        self._reader: easyocr.Reader = load_or_build_object(
            easyocr.Reader, self.lang, gpu=self.device)

    def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('OCR results, include bbox in x1, y1, x2, y2 format '
                        'and the recognized text.')]:

        image = image.to_array()
        ocr_results = self._reader.readtext(image, detail=1, **self.read_args)
        outputs = []
        for result in ocr_results:
            bbox = self.extract_bbox(result[0])
            pred = result[1]
            outputs.append('({}, {}, {}, {}) {}'.format(*bbox, pred))
        outputs = '\n'.join(outputs)
        return outputs

    @staticmethod
    def extract_bbox(char_boxes) -> Tuple[int, int, int, int]:
        xs = [int(box[0]) for box in char_boxes]
        ys = [int(box[1]) for box in char_boxes]
        return min(xs), min(ys), max(xs), max(ys)
