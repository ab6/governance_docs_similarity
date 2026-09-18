# Datasheet: Emerging Technology Observatory AGORA dataset

**Version**: 1.33.0
**Generated**: 2026-09-18 17:09 UTC

## Motivation

Updated regularly, AGORA includes summaries, document text, thematic tags, and filters to help you quickly discover and analyze key developments in AI governance.The AGORA dataset consists of the csv tables documents, segments, collections, and authorities and a folder of text documents called fulltext.

## Composition

- **Total size**: 1251 text files; 4 csv tables with 1251, 1251, 1251, and 1251 rows respectively.
- **Source**: It is a collection of AI-relevant laws, regulations, standards, and other governance documents from the United States and around the world.
- **Collection method**: Document text and some metadata are taken from official sources. Other metadata, summaries, and tags are produced by ETO analysts and annotators.

### Features

| Feature | Type | Description |
|---------|------|-------------|
| documents | csv | Contains metadata and summaries of governance documents. |
| segments | csv | Contains segments of text extracted from the documents. |
| collections | csv | Contains information about collections of documents. |
| authorities | csv | Contains information about authorities responsible for the documents. |
| fulltext | folder of text files | Contains the full text of the governance documents in individual text files. |


### Demographic Composition

No sensitive information such as personally identifiable information (PII) or protected attributes (e.g., race, gender, age) is included in the dataset.

### Known Biases and Limitations

The current dataset skews toward U.S. law and policy.

## Preprocessing

- {'annotation': 'Summaries, tags, and metadata are produced by LLM, ETO analysts and annotators.'}

## Uses

### Intended Use

AGORA fulltext files are unofficial copies of the underlying documents, collected by our screeners for research purposes and users' convenience.

### Prohibited Use

This dataset is subject to ETO's general terms of use. If you use it, please cite us. Note that the dataset includes full text of AGORA documents, taken from sources such as government websites and repositories. Given its nature, we believe all of this material is open to non-commercial use consistent with our general terms of use, but we make no warranties.

## Distribution and Maintenance

- **Retention policy**: Dataset retained indefinitely. AGORA is updated regularly, and new versions of the dataset are released periodically.
- **Contact**: cset@georgetown.edu

## Citation

Datasheet generated following the framework proposed by
[Gebru et al., 2021](https://arxiv.org/abs/1803.09010).
