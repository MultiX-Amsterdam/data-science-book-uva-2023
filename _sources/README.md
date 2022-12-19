# data-science-book-uva

This repository hosts the website for the data science course (Bachelor level) at the Informatics Institute, University of Amsterdam. The webite is built using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

Below are the steps to update and build this book. First, clone this repository to your local machine. Assume that you already have [miniconda](https://docs.conda.io/en/main/miniconda.html) installed. Next, install the jupyter-book package:
```sh
$ conda create -n jupyterbook
$ conda activate jupyterbook
$ conda install python
$ pip install -U jupyter-book
$ pip install ghp-import
$ jupyter-book --help
```
Then, clone this repository and build the book:
```sh
$ git clone https://github.com/MultiX-Amsterdam/data-science-book-uva
$ cd data-science-book-uva
$ jupyter-book build .
```
Once it is done, you can view the book in the [html content](_build/html) folder using a web browser. To update the book online in this GitHub repository (in the gh-pages branch), run the following:
```sh
$ ghp-import -n -p -f _build/html
```
The above steps will update the gh-pages branch, which hosts the website. Finally, follow the normal git flow to commit the changes and push the code to the main branch in this repository.
