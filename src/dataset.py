import torch
from torch.utils.data import Dataset
from augmentex import WordAug, CharAug

class AugmentedIntentDataset(Dataset):
    """
    Dataset for intent classification with automatic augmentation.
    Supports augmentations: synonym_replacement, back_translation, paraphrasing, morphological.
    """
    def __init__(self, texts, labels, tokenizer, augment=False, augment_kwargs=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_kwargs = augment_kwargs or {}

        self.word_aug = WordAug(
            unit_prob=0.4,
            min_aug=1,
            max_aug=5,
            lang="rus",
            platform="pc",
            random_seed=42
        )

        self.char_aug = CharAug(
            unit_prob=0.3,
            min_aug=1,
            max_aug=5,
            mult_num=3,
            lang="rus",
            platform="pc",
            random_seed=42
        )

    def __len__(self):
        return len(self.texts)

    def _apply_augmentation(self, text):
        augmentations = self.augment_kwargs.get("augmentations", [])
        for aug_type in augmentations:
            if aug_type == "synonym_replacement":
                text = self.word_aug.augment(text, action="replace")
            elif aug_type == "morphological":
                text = self.word_aug.augment(text, action="morph")
            elif aug_type == "typo":
                text = self.char_aug.augment(text, action="typo")
            elif aug_type == "random_char_insert":
                text = self.char_aug.augment(text, action="insert")
        return text

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.augment:
            text = self._apply_augmentation(text)

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }