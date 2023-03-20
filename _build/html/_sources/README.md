# data-science-book-uva

This repository hosts the website for the data science course (Bachelor level) at the Informatics Institute, University of Amsterdam. The webite is built using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

All the course content in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

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
To rebuild the entire book, use the following:
```sh
$ jupyter-book build --all .
```
Once it is done, you can view the book in the [html content](_build/html) folder using a web browser. To update the book online in this GitHub repository (in the gh-pages branch), run the following:
```sh
$ ghp-import -n -p -f _build/html
```
The above steps will update the gh-pages branch, which hosts the website. Finally, follow the normal git flow to commit the changes and push the code to the main branch in this repository.

## Error Handling

You many encounter an error that looks like the following when building the notebook:
```sh
Extension error (sphinx_external_toc.events):
Handler <function add_changed_toctrees at 0x10a031750> for event 'env-get-outdated' threw an exception (exception: 'Document' object has no attribute 'docname')
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniconda/base/envs/jupyterbook/lib/python3.10/site-packages/sphinx/events.py", line 94, in emit
    results.append(listener.handler(self.app, *args))
  File "/opt/homebrew/Caskroom/miniconda/base/envs/jupyterbook/lib/python3.10/site-packages/sphinx_external_toc/events.py", line 138, in add_changed_toctrees
    filenames = site_map.get_changed(previous_map)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/jupyterbook/lib/python3.10/site-packages/sphinx_external_toc/api.py", line 220, in get_changed
    if self.root.docname != previous.root.docname:
AttributeError: 'Document' object has no attribute 'docname'
```
In this case, just run the following command, and the problem should disappear:
```sh
pip install jupyter-cache
```
