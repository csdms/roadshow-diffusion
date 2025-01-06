# roadshow-diffusion

A model used for teaching in the [CSDMS Roadshow](https://csdms.colorado.edu/wiki/Roadshows).

## What is CSDMS and why are we here?

[CSDMS](https://csdms.colorado.edu) is an international community of researchers studying surface processes, with an emphasis on modeling.
Modeling requires code.
Historically, scientists haven't been great at writing code that's maintainable and easily used by others.
Therefore, a secondary emphasis at CSDMS is on helping scientists write [FAIR software](https://www.nature.com/articles/s41597-022-01710-x)--code that's findable, accessible, interoperable, and reusable.
That's why we're here.
Science is first, but software is an engine that drives science.
Better software enhances scientific productivity.

## What are we going to do?

We're going to develop a model.
We'll use a technique a lot of grad students we know use.
Along the way, we'll comment on best practices in geoscientific software development.

Topics we'll encounter:

* Shell commands
* Modularization
* Version control
* Text editors
* Virtual environments
* Package management
* Unit testing
* Continuous integration
* Documentation

If we have time, we'll show a more comprehensive workflow for developing a model.

We'll finish by working with [Landlab](https://landlab.csdms.io/), a toolkit for developing models.
It takes care of many of the tedious details that a scientist would have to deal with when developing a model.

## How will this work?

Here are the topics we'll cover in order to develop our model.

* Project Jupyter
    * JupyterHub: login to explore Hub
    * JupyterLab: show components
    * Notebook: open a new notebook and show basics
* Shell (bash) commands
* Diffusion model in a notebook I
* Intro to Git/GitHub
    * Set up SSH keys
    * Set up a repo for the diffusion model notebook
* Diffusion model in a notebook II
    * Refactor, modularize model in notebook
* Export notebook to Python source
* Text editors
* Virtual environments
    * conda
    * venv or virtualenv
* Refactor diffusion model in Python source
    * Rename file to diffusion.py to adhere to module naming rules
* Package model
    * Module definition file
    * Basic pyproject.toml file
    * Show how to pip install into a venv
* Unit tests
* Documentation
* Diffusion model in a notebook III
    * Import diffusion model from new package
* Visualize with Jupyter widgets
* Landlab
