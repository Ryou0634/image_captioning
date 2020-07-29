from typing import Dict, List
from collections import defaultdict
import tqdm
import re
import numpy as np

from allennlp.data import DatasetReader
from allennlp.data import Token, Instance
from allennlp.data.fields import TextField, ArrayField, MetadataField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import logging

logger = logging.getLogger(__name__)


def read_flickr_caption_file(caption_file_path: str, delimiter: str = "\t") -> Dict[str, List[str]]:
    logger.info(f"Reading captions from {caption_file_path}...")

    image_caption_dict = defaultdict(list)

    with open(caption_file_path, "r") as f:
        for line in tqdm.tqdm(f):
            image_file_tag, caption = re.split(delimiter, line.strip(), maxsplit=1)
            image_file_name = image_file_tag.split("#")[0]
            image_caption_dict[image_file_name].append(caption)

    return image_caption_dict


def read_image_feature_files(image_feature_path: str, image_feature_filename_path: str) -> Dict[str, np.ndarray]:
    features = np.load(image_feature_path)
    with open(image_feature_filename_path, "r") as f:
        image_filenames = f.read().strip().split("\n")
    return dict(zip(image_filenames, features))


@DatasetReader.register("flickr")
class FlickrReader(DatasetReader):
    def __init__(self, caption_file_path: str, image_feature_path: str, image_feature_filename_path: str):
        super().__init__(lazy=False)
        self.captions = read_flickr_caption_file(caption_file_path)
        self.image_features = read_image_feature_files(image_feature_path, image_feature_filename_path)
        self._token_indexers = {
            "tokens": SingleIdTokenIndexer(
                namespace="tgt_tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL]
            )
        }

    def _read(self, file_path: str):
        with open(file_path, "r") as f:
            for line in f:
                filename = line.strip()
                yield self.text_to_instance(
                    image_feature=self.image_features[filename], captions=self.captions[filename], image_name=filename
                )

    def text_to_instance(self, image_feature: np.ndarray, captions: List[str] = None, image_name: str = None):

        fields = {
            "image_feature": ArrayField(image_feature),
        }

        if captions is not None:
            text_field_list = []
            for caption in captions:
                tokens = [Token(t) for t in caption.split()]
                text_field_list.append(TextField(tokens, self._token_indexers))
            fields["target_tokens"] = ListField(text_field_list)

        if image_name:
            fields["image_name"] = MetadataField(image_name)

        return Instance(fields)
