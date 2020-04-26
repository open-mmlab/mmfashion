#PolyvoreOutfitDataset

Polyvore Dataset is widely used for learning fashion compatibility.
There are several different split versions of this dataset. We deployed [Learning Type Aware Embeddings for Fashion Compatibility](https://arxiv.org/pdf/1803.09196.pdf) here.

- Polyvore Outfits(Nondisjoint)

    Outfits are split at random. The training dataset and testing dataset may have overlapped items(garments).

- Polyvore Outfits(Disjoint)

    The training set and testing/validation set have no overlapped items.

## Annotations

- train/test/valid.json

    A list of outfits, their item_id and their ordering(index).

- typespaces.p

    A list of tuples(t1, t2), each of which identifies a type- specific embedding that compares items of type t1 to items of type t2.

- train_hglmm_pca6000.txt

    Each row contains 6001 comma separated values, where the first element is the label, the remaining 6000 dimensions are the PCA-reduced HGLMM fisher vectors (note: the "label" may also contain a comma).

- compatibility_train/test/valid.txt

    Fashion compatibility experiment data, where each row is an outfit sample. The first element of the outfit sample is the label (1/0 for positive/negative) and the remaining elements are item identifiers in that sample.

- fill_in_blank_train/test/valid.json

    Fill-in-the-blank experiment data, contains an array of dictionaries. These dictionaries contain the question/answer pairs, and also identifies the "index" of the item in the outfit in
"blank_position".  Since the set_id is used in the item identifiers, the correct answer can be determined by matching the set_id in the question
elements with the set_id in the answers.


##Images
Images are stored by their item_id, which are organized in lists of outfits for each each version of the dataset.

##Meta-Data
- polyvore_item_metadata.json

    It contains a dictionary where each key is an item_id, and the values are its associated meta-data labels.

- polyvore_outfit_titles.json

    It contains a dictionary where each key is a set_id and the values are its associated meta-data labels.

- categories.csv

    Each row contains three items: category_id, fine_grained category, semantic categroy.
