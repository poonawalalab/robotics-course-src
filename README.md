# Robotics Website

Contains the quarto files for generating the website.

Also contains the code for examples that are available:

1. On the website
1. Offline to test examples

## Setup 



## Local development loop

Start in `dlcourse-site-src/`
```
git pull          # in the private repo
quarto render     # writes HTML into ./_dlcourse
cd _dlcourse
git add .
git commit -m "update site"
git push          # pushes to *public* repo
cd ..
git add .gitmodules _dlcourse
git commit -m "update submodule pointer"
git push          # pushes to *private* repo
```
