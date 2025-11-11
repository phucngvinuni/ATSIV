---
task_categories:
- image-classification
- image-segmentation
tags:
- fish
- traits
- processed
- RGB
- biology
- image
- animals
- CV
pretty_name: Fish-Vista
size_categories:
- 10K<n<100K
language:
- en
configs:
- config_name: species_classification
  data_files:
  - split: train
    path: classification_train.csv
  - split: test
    path: classification_test.csv
  - split: val
    path: classification_val.csv
- config_name: species_trait_identification
  data_files:
  - split: train
    path: identification_train.csv
  - split: test_insp
    path: identification_test_insp.csv
  - split: test_lvsp
    path: identification_test_lvsp.csv
  - split: val
    path: identification_val.csv
- config_name: trait_segmentation
  data_files:
  - "segmentation_data.csv"
  - "segmentation_masks/images/*.png"
---

<!--
Image with caption:
|![Figure #](https://huggingface.co/imageomics/datasets/<data-repo>/resolve/main/<filename>)|
|:--|
|**Figure #.** [Image of <>](https://huggingface.co/datasets/imageomics/<data-repo>/raw/main/<filename>) <caption description>.|
-->

# Dataset Card for Fish-Visual Trait Analysis (Fish-Vista)
* Note that the '**</Use this dataset>**' option will only load the CSV files. To download the entire dataset, including all processed images and segmentation annotations, refer to [Instructions for downloading dataset and images](https://huggingface.co/datasets/imageomics/fish-vista#instructions-for-downloading-dataset-and-images).
* See [Example Code to Use the Segmentation Dataset])(https://huggingface.co/datasets/imageomics/fish-vista#example-code-to-use-the-segmentation-dataset)
|![Figure 1](https://huggingface.co/datasets/imageomics/fish-vista/resolve/main/metadata/figures/FishVista.png)|
|:--|
|**Figure 1.** A schematic representation of the different tasks in Fish-Vista Dataset. |

## Instructions for downloading dataset and images
<!-- [Add instructions for downloading images here]
-->
* Install [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
* Git clone the fish-vista repository
  * Run the following commands in a **terminal**:
```bash
git clone https://huggingface.co/datasets/imageomics/fish-vista
cd fish-vista
```
* Run the following commands to move all chunked images to a single directory:
```bash
mkdir AllImages
find Images -type f -exec mv -v {} AllImages \;
rm -rf Images
mv AllImages Images
```

  * You should now have all the images in the *Images* directory

* Install requirements.txt
```bash
pip install -r requirements.txt
```
* Run the following commands to download and process copyrighted images 
```bash
python download_and_process_nd_images.py --save_dir Images
```
  * This will download and process the CC-BY-ND images that we do not provide in the *Images* folder

## Dataset Structure

```
/dataset/
    segmentation_masks/
        annotations/
        images/
    Images/
        chunk_1
          filename 1
          filename 2
          ...
          filename 10k
        chunk_2
          filename 1
          filename 2
          ...
          filename 10k
        .
        .
        .
        chunk_6
          filename 1
          filename 2
          ...
          filename 10k
    ND_Processing_Files
    download_and_process_nd_images.py
    classification_train.csv
    classification_test.csv
    classification_val.csv
    identification_train.csv
    identification_test.csv
    identification_val.csv
    segmentation_data.csv
    segmentation_train.csv
    segmentation_test.csv
    segmentation_val.csv
    metadata/
        figures/
            # figures included in README
        data-bib.bib
```


### Data Instances

<!-- Add information about each of these (task, number of images per split, etc.). Perhaps reformat as <task>_<split>.csv.
-->

* **Species Classification (FV-419):** `classification_<split>.csv`
  * Approximately 48K images of 419 species for species classification tasks.
  * There are about 35K training, 7.6K test, and 5K validation images.

* **Trait Identification (FV-682):** `identification_<split>.csv`
  * Approximately 53K images of 682 species for trait identification based on _species-level trait labels_ (i.e., presence/absence of traits based on trait labels for the species from information provided by [Phenoscape]() and [FishBase](https://www.fishbase.se/)).
  * About 38K training, 8K `test_insp` (species in training set), 1.9K `test_lvsp` (species not in training), and 5.2K validation images.
  * Train, test, and validation splits are generated based on traits, so there are 628 species in train, 450 species in `test_insp`, 51 species in `test_lvsp`, and 451 in the validation set (3 species only in val). 

* **Trait Segmentation (FV-1200):** `segmentation_<split>.csv` 
  * Pixel-level annotations of 9 different traits for 2,427 fish images.
  * About 1.7k training, 600 test and 120 validation images for the segmentation task
  * These are also used as manually annotated test set for Trait Identification.

* **All Segmentation Data:** `segmentation_data.csv` 
  * Essentially a collation of the trait segmentation splits
  * Used for evaluating trait identification on the entire FV-1200


* **Image Information**
  * **Type:** JPG
  * **Size (x pixels by y pixels):** Variable
  * **Background (color or none):** Uniform (White)


### Data Fields

CSV Columns are as follows:

- `filename`: Unique filename for our processed images.
- `source_filename`: Filename of the source image. Non-unique, since one source filename can result in multiple crops in our processed dataset.
- `original_format`: Original format, all jpg/jpeg.
- `arkid`: ARKID from FishAIR for the original images. Non-unique, since one source file can result in multiple crops in our processed dataset.
- `family`: Taxonomic family
- `source`: Source museum collection. GLIN, Idigbio or Morphbank
- `owner`: Owner institution within the source collection.
- `standardized_species`: Open-tree-taxonomy-resolved species name. This is the species name that we provide for Fish-Vista
- `original_url`: URL to download the original, unprocessed image
- `file_name`: Links to the image inside the repository. Necessary for HF data viewer. Not to be confused with `filename`
- `license`: License information for the original image
- `adipose_fin`: Presence/absence of the adipose fin trait. NA for the classification (FV-419) dataset, since it is only used for identification. 1 indicates presence and 0 indicates absence. This is used for trait identification.
- `pelvic_fin`: Presence/absence of the pelvic trait. NA for the classification (FV-419) dataset, since it is only used for identification. 1 indicates presence and 0 indicates absence. This is only used for trait identification.
- `barbel`: Presence/absence of the barbel trait. NA for the classification (FV-419) dataset, since it is only used for identification. 1 indicates presence and 0 indicates absence. This is used for trait identification.
- `multiple_dorsal_fin`: Presence/absence of the dorsal fin trait. NA for the classification (FV-419) dataset, since it is only used for identification. 1 indicates presence, 0 indicates absence and -1 indicates unknown. This is used for trait identification.



### Data Splits

For each task (or subset), the split is indicated by the CSV name (e.g., `classification_<split>.csv`). More information is provided in [Data Instances](#data-instances), above.

## Example Code to Use the Segmentation Dataset

We provide an example code to use the FV-1200 segmentation dataset for convenience of users. Please install *pillow*, *numpy*, *pandas* and *matplotlib* before trying the code:

```python
from PIL import Image
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Set the the fish_vista_repo_dir to the path of your cloned fish-vista HF repository. This code assumes you are running from within the fish-vista directory
fish_vista_repo_dir = '.'

# segmentation_masks/images contains the annotated segmentation maps for the traits. 
# If image filename is <image_filename>.jpg, the corresponding annotation is contained in segmentation_masks/images/<image_filename>.png 
seg_mask_path = os.path.join(fish_vista_repo_dir, 'segmentation_masks/images')

# seg_id_trait_map.json maps the annotation id to the corresponding trait name.
# For example, pixels annotated with 1 correspond to the trait: 'Head'
id_trait_map_file = os.path.join(fish_vista_repo_dir, 'segmentation_masks/seg_id_trait_map.json')
with open(id_trait_map_file, 'r') as f:
    id_trait_map = json.load(f)

# Read a segmentation csv file
train_path = os.path.join(fish_vista_repo_dir, 'segmentation_train.csv')
train_df = pd.read_csv(train_path)

# Get image and segmentation mask of image at index 'idx'
idx = 0
img_filename = train_df.iloc[idx].filename
img_mask_filename = os.path.splitext(img_filename)[0]+'.png'
# Load and view the mask 
img_mask = Image.open(os.path.join(seg_mask_path, img_mask_filename))
plt.imshow(img_mask)

# List the traits that are present in this image
img_mask_arr = np.asarray(img_mask)
print([id_trait_map[str(trait_id)] for trait_id in np.unique(img_mask_arr)])
```

## Dataset Details

### Dataset Description

<!--
- **Curated by:** list curators (authors for _data_ citation, moved up)
- **Language(s) (NLP):** [More Information Needed]
<!-- Provide the basic links for the dataset. These will show up on the sidebar to the right of your dataset card ("Curated by" too). -->
<!--
- **Homepage:** 
- **Repository:** [related project repo]
- **Paper:** 
-->

<!-- Provide a longer summary of what this dataset is. -->

The Fish-Visual Trait Analysis (Fish-Vista) dataset is a large, annotated collection of 60K fish images spanning 1900 different species; it supports several challenging and biologically relevant tasks including species classification, trait identification, and trait segmentation. These images have been curated through a sophisticated data processing pipeline applied to a cumulative set of images obtained from various museum collections. Fish-Vista provides fine-grained labels of various visual traits present in each image. It also offers pixel-level annotations of 9 different traits for 2427 fish images, facilitating additional trait segmentation and localization tasks.

The Fish Vista dataset consists of museum fish images from [Great Lakes Invasives Network (GLIN)](https://greatlakesinvasives.org/portal/index.php), [iDigBio](https://www.idigbio.org/), and [Morphbank](https://www.morphbank.net/) databases. We acquired these images, along with associated metadata including the scientific species names, the taxonomical family the species belong to, and licensing information, from the [Fish-AIR repository](https://fishair.org/). 


<!--This dataset card has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1), and further altered to suit Imageomics Institute needs.-->


### Supported Tasks and Leaderboards
<!--[Add some more description. could replace graphs with tables]-->

|![Figure 2](https://huggingface.co/datasets/imageomics/fish-vista/resolve/main/metadata/figures/clf_imbalance.png)|
|:--|
|**Figure 2.** Comparison of the fine-grained classification performance of different imbalanced classification methods. |

|![Figure 3](https://huggingface.co/datasets/imageomics/fish-vista/resolve/main/metadata/figures/IdentificationOriginalResults.png)|
|:--|
|**Figure 3.** Trait identification performance of different multi-label classification methods. |


<!---
This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1).
--->

### Languages

English

## Dataset Creation

### Curation Rationale
<!-- Motivation for the creation of this dataset. For instance, what you intended to study and why that required curation of a new dataset (or if it's newly collected data and why the data was collected (intended use)), etc. -->

Fishes are integral to both ecological systems and economic sectors, and studying fish traits is crucial for understanding biodiversity patterns and macro-evolution trends.
Currently available fish datasets tend to focus on species classification. They lack finer-grained labels for traits. When segmentation annotations are available in existing datasets, they tend to be for the entire specimen, allowing for segmenation of background, but not trait segmentation. 
The ultimate goal of Fish-Vista is to provide a clean, carefully curated, high-resolution dataset that can serve as a foundation for accelerating biological discoveries using advances in AI.


### Source Data

<!-- This section describes the source data (e.g., news text and headlines, social media posts, translated sentences, ...). As well as an original source it was created from (e.g., sampling from Zenodo records, compiling images from different aggregators, etc.) -->
Images and taxonomic labels were aggregated by [Fish-AIR](https://fishair.org/) from 
- [Great Lakes Invasives Network (GLIN)](https://greatlakesinvasives.org/portal/index.php)
- [iDigBio](https://www.idigbio.org/)
- [Morphbank](https://www.morphbank.net/)
- [Illinois Natural History Survey (INHS)](https://biocoll.inhs.illinois.edu/portal/index.php)
- [Minnesota Biodiversity Atlas, Bell Museum](https://bellatlas.umn.edu/index.php)
- [University of Michigan Museum of Zoology (UMMZ), Division of Fishes](https://ipt.lsa.umich.edu/resource?r=ummz\_fish)
- [University of Wisconsin-Madison Zoological Museum - Fish](http://zoology.wisc.edu/uwzm/)
- [Field Museum of Natural History (Zoology, FMNH) Fish Collection](https://fmipt.fieldmuseum.org/ipt/resource?r=fmnh_fishes)
- [The Ohio State University Fish Division, Museum of Biological Diversity (OSUM), Occurrence dataset](https://doi.org/10.15468/subsl8)

[Phenoscape](https://kb.phenoscape.org/about/phenoscape/kb) and [FishBase](https://www.fishbase.se/search.php) were used to provide the information on traits at the species level.

[Open Tree Taxonomy](https://tree.opentreeoflife.org/) was used to standardize the species names provided by Fish-AIR.


#### Data Collection and Processing
<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, re-sizing of images, tools and libraries used, etc. 
This is what _you_ did to it following collection from the original source; it will be overall processing if you collected the data initially.
-->

|![Figure 4](https://huggingface.co/datasets/imageomics/fish-vista/resolve/main/metadata/figures/DataProcessingPipelineFishVista.png)|
|:--|
|**Figure 4.** An overview of the data processing and filtering pipeline used to obtain Fish-Vista. |

We carefully curated a set of
60K images sourced from various museum collections through [Fish-AIR](https://fishair.org/), including [Great Lakes Invasives Network (GLIN)](https://greatlakesinvasives.org/portal/index.php), [iDigBio](https://www.idigbio.org/), and [Morphbank](https://www.morphbank.net/).
Our pipeline incorporates rigorous stages such as duplicate removal, metadata-driven filtering, cropping, background removal using the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), and a final
manual filtering phase. Fish-Vista supports several biologically meaningful tasks such as species
classification, trait identification, and trait segmentation. 



### Annotations
<!-- 
If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. 

Ex: We standardized the taxonomic labels provided by the various data sources to conform to a uniform 7-rank Linnean structure. (Then, under annotation process, describe how this was done: Our sources used different names for the same kingdom (both _Animalia_ and _Metazoa_), so we chose one for all (_Animalia_). -->

#### Annotation process
<!-- This section describes the annotation process such as annotation tools used, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->
[Phenoscape](https://kb.phenoscape.org/about/phenoscape/kb) and [FishBase](https://www.fishbase.se/search.php) were used to provide the information on species-level traits (the species-trait matrix).

[Open Tree Taxonomy](https://tree.opentreeoflife.org/) was used to standardize the species names provided by Fish-AIR.

Image-level trait segmentations were manually annotated as described below.

The annotation process for the segmentation subset was led by Wasila Dahdul. She provided guidance and oversight to a team of three people from [NEON](https://www.neonscience.org/about), who used [CVAT](https://zenodo.org/records/7863887) to label nine external traits in the images. These traits correspond to the following terms for anatomical structures in the UBERON anatomy ontology:
1. Eye, [UBERON_0000019](http://purl.obolibrary.org/obo/UBERON_0000019)
2. Head, [UBERON_0000033](http://purl.obolibrary.org/obo/UBERON_0000033)
3. Barbel, [UBERON_2000622](http://purl.obolibrary.org/obo/UBERON_2000622)
4. Dorsal fin, [UBERON_0003097](http://purl.obolibrary.org/obo/UBERON_0003097)
5. Adipose fin, [UBERON_2000251](http://purl.obolibrary.org/obo/UBERON_2000251)
6. Pectoral fin, [UBERON_0000151](http://purl.obolibrary.org/obo/UBERON_0000151)
7. Pelvic fin, [UBERON_0000152](http://purl.obolibrary.org/obo/UBERON_0000152)
8. Anal fin, [UBERON_4000163](http://purl.obolibrary.org/obo/UBERON_4000163)
9. Caudal fin, [UBERON_4000164](http://purl.obolibrary.org/obo/UBERON_4000164)


### Personal and Sensitive Information

None

## Considerations for Using the Data

### Discussion of Biases and Other Known Limitations

- This dataset is imbalanced and long tailed 
- It inherits biases inherent to museum images
- Train sets may contain noisy images (in very small numbers)



### Recommendations
<!--[More Information Needed]
 This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

## Licensing Information
<!--[More Information Needed]

 See notes at top of file about selecting a license. 
If you choose CC0: This dataset is dedicated to the public domain for the benefit of scientific pursuits. We ask that you cite the dataset and journal paper using the below citations if you make use of it in your research.

Be sure to note different licensing of images if they have a different license from the compilation.
ex: 
The data (images and text) contain a variety of licensing restrictions mostly within the CC family. Each image and text in this dataset is provided under the least restrictive terms allowed by its licensing requirements as provided to us (i.e, we impose no additional restrictions past those specified by licenses in the license file).

EOL images contain a variety of licenses ranging from [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).
For license and citation information by image, see our [license file](https://huggingface.co/datasets/imageomics/treeoflife-10m/blob/main/metadata/licenses.csv).

This dataset (the compilation) has been marked as dedicated to the public domain by applying the [CC0 Public Domain Waiver](https://creativecommons.org/publicdomain/zero/1.0/). However, images may be licensed under different terms (as noted above).
-->

The source images in our dataset come with various licenses, mostly within the Creative Commons family. We provide license and citation information, including the source institution for each image, in our metadata CSV files available in the HuggingFace repository. Additionally, we attribute each image to the original FishAIR URL from which it was downloaded.

A small subset of our images (approximately 1k) from IDigBio are licensed under CC-BY-ND, which prohibits us from distributing processed versions of these images. Therefore, we do not publish these 1,000 images in the repository. Instead, we provide the URLs for downloading the original images and a processing script that can be applied to obtain the processed versions we use.

Our dataset is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). However, individual images within our dataset may have different licenses, which are specified in our CSV files.

## Citation
<!--[More Information Needed]
-->

If you use Fish-Vista in your research, please cite both our paper and the dataset.
Please be sure to also cite the original data sources using the citations provided in [metadata/data-bib.bib](https://huggingface.co/datasets/imageomics/fish-vista/blob/main/metadata/data-bib.bib).


**BibTeX:**

**Paper**
```
@misc{mehrab2024fishvista,
      title={Fish-Vista: A Multi-Purpose Dataset for Understanding & Identification of Traits from Images}, 
      author={Kazi Sajeed Mehrab and M. Maruf and Arka Daw and Harish Babu Manogaran and Abhilash Neog and Mridul Khurana and Bahadir Altintas and Yasin Bakis and Elizabeth G Campolongo and Matthew J Thompson and Xiaojun Wang and Hilmar Lapp and Wei-Lun Chao and Paula M. Mabee and Henry L. Bart Jr. au2 and Wasila Dahdul and Anuj Karpatne},
      year={2024},
      eprint={2407.08027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08027}, 
}
```


**Data**
```
@dataset{fishvistaData,
  title    = {Fish-Vista: A Multi-Purpose Dataset for Understanding & Identification of Traits from Images},
  author   = {Kazi Sajeed Mehrab and M. Maruf and Arka Daw and Harish Babu Manogaran and Abhilash Neog and Mridul Khurana and Bahadir Altintas and Yasin Bakış and Elizabeth G Campolongo and Matthew J Thompson and Xiaojun Wang and Hilmar Lapp and Wei-Lun Chao and Paula M. Mabee and Henry L. Bart Jr. and Wasila Dahdul and Anuj Karpatne},
  year     = {2024},
  url      = {https://huggingface.co/datasets/imageomics/fish-vista},
  doi      = {10.57967/hf/3471},
  publisher = {Hugging Face}
}
```



## Acknowledgements

This work was supported by the [Imageomics Institute](https://imageomics.org), which is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
We would like to thank Shelley Riders, Jerry Tatum, and Cesar Ortiz and for segmentation data annotation.
<!-- You may also want to credit the source of your data, i.e., if you went to a museum or nature preserve to collect it. -->

## Glossary 

<!-- [optional] If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

## More Information 

<!-- [optional] Any other relevant information that doesn't fit elsewhere. -->

## Dataset Card Authors 

Kazi Sajeed Mehrab and Elizabeth G. Campolongo

## Dataset Card Contact

<!--[More Information Needed--optional]
 Could include who to contact with questions, but this is also what the "Discussions" tab is for. -->
 ksmehrab@vt.edu
