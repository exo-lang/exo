
## Notes for Installation and Dependencies

We make active use of newer Python 3.x features such as f-strings, so please use a more recent (I am writing this in Sept. 2020) version of Python if you're getting errors about unsupported features.

### Requirements.txt

We are trying to maintain a dependency listing in requirements.txt.  You should be able to install all of these dependencies with the command
```
pip install -r requirements.txt
```

### PySMT

Depending on your setup, getting PySMT to work correctly may be difficult.  You need to independently install a solver such as Z3 or CVC4, and even then getting the PySMT library to correctly locate that solver may be difficult.  We have included the `z3-solver` python package as a requirement, which will hopefully avoid this issue, but you can also install z3 (or your choice of solver) independently.

### Submodules
If you have an ssh key uploaded to GitHub, you should be able to simply run the "it handles everything" command
```
git submodule update --init --recursive
```

If you do not have an ssh key uploaded and want to change the submodule URL to use (e.g.) HTTPS, then try the [following scheme](https://stackoverflow.com/questions/42028437/how-to-change-git-submodules-url-locally/42035018#:~:text=If%20you%20want%20to%20modify,that%20you%20want%20to%20push.&text=Then%20modify%20the%20.,the%20submodule%20URL%20as%20usual.)

First initialize the submodule configuration
```
git submodule init
```

Then go to `.git/config` (NOT `.gitmodules`) and edit the URL (for the submodule) to the form you want.

Finally run
```
git submodule update --recursive
```
to pull through the new URL and establish the link.


### PLEASE Set Up The Hooks

We cannot automate the enabling of git hooks on your copy of the repository, but the following is the closest we can come to that.  After you clone the repository, please run the following command in the root directory for the project
```
git config core.hooksPath githooks
```

This will setup the repository to use the provided project hooks.  In particular, we have pre-commit hooks there that will prevent the repository history getting bloated with iPython Notebook output.


## Notes for Testing

To run the tests, simply type
```
pytest
```
in the root of the project

### Running Coverage Testing

To run pytest with coverage tests, execute
```
pytest --cov=SYS_ATL tests/
```
Then, if you want to see annotated source files, run
```
coverage html
```

