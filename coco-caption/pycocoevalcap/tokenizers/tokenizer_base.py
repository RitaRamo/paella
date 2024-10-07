class BaseTokenizer:
    """A base dummy tokenizer to derive from."""

    def signature(self):
        """
        Returns a signature for the tokenizer.

        :return: signature string
        """
        return 'none'

    def __call__(self, line):
        """
        Tokenizes an input line with the tokenizer.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return line


    def tokenize(self, captions_for_image):
        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
        sentences = ([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])

        for k, line in zip(image_id, sentences):
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = self.__call__(line.lower().rstrip())
            
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image