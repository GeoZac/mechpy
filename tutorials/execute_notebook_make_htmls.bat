:: change console to the current working directory
Pushd "%~dp0"

::loops for all *.ipynb files 
for %%i in (*.ipynb) do jupyter nbconvert --execute %%i

