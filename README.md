mechpy - a mechanical engineer's python toolbox

Tutorials - see the [nbviewer for mechpy](http://nbviewer.jupyter.org/github/nagordon/mechpy/blob/master/mechpy.ipynb)

Mechpy was created for two reasons. 
 * To provide the practicing engineer with applications or boilerplate code to quickly replicate and solve real-world systems common in mechanical engineering
 * To give the engineering student a code baes from which to suppliment learning through hand-calculations and an easy way to check work.

There are many different tools available to engineers. Hand-calcsulations, spreadsheets, and code are all great ways to perform calculations or visualize data or concepts. MATLAB is the defacto tool to solve many engineering calulations, but is just too expensive to be a practical tool for many of us. Octave, Scilab, or Freelab are great alternatives, but is limited in scope to calculation. I began using python for calculations and visualzations and have found it to be a very powerful tool with many existing modules for mathematics and plotting, in addition to the thousands of other libraries for general computing.

TODO
* clean up stupid links 
* check out FEM - Numerical Python pg 690/1311
* add failure criteria to composites and mutli-material
* add composite plate code to composites
* convert matlab code for plotting test data

https://github.com/jrjohansson/scientific-python-lectures
https://github.com/numerical-mooc/numerical-mooc

- - - -

## Modules

### abaqus and abaqus_report
API code to automate running of finite element models and reporting using python-pptx 

### catia
API code to automate cad in catia 

### composites  
specialized code to analyze composite plates using classical laminated plate theory

### design  
shear-bending diagrams, assisting hand calculations for stress

### dynamics  

### fem  
example code for computing finite element models for analysis and education

### math  
various math tools for processing engineering systems

### statics  
code for assisting and checking hand calculations

### units  
unit conversion

- - - -

## References
Hibbler - Statics  
Hibbler - Mechanics of Materials  
Collins et al - Mechanical Design of Machine Elements and Machines
Flabel - Practical Stress Analysis for Design Engineers
Peery - Aircraft Structures
Niu - Airframe Stress Analysis and Sizing
[Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) with [source code](http://www.apress.com/downloadable/download/sample/sample_id/1732/)  
[Elementary Mechanics Using Python](http://www.springer.com/us/book/9783319195957#aboutBook)  
A Primer on Scientific Programming With Python  

- - - - 
## Python Engineering Libraries
Check out the dependencies of mechpy for specific examples and documentation
 * [Scipy](), [scipy cookbook](http://scipy-cookbook.readthedocs.io/index.html   )
   * numpy
   * sympy
   * matplotlib
   * pandas
 * [pyndamics](https://github.com/bblais/pyndamics  ) with [example](http://nbviewer.ipython.org/gist/bblais/7321928)
 * [Unum units](https://pypi.python.org/pypi/Unum)
 * [pint units](http://pint.readthedocs.io/en/0.7.2/  )
 * [pynastran](https://github.com/SteveDoyle2/pynastran/wiki/GUI)
 * [aeropython](https://github.com/barbagroup/AeroPython)
 * [control-systems](https://github.com/python-control/python-control)
 * [grid solvers](http://pyamg.org/) with [example)[https://code.google.com/p/pyamg/wiki/Examples]
 * [python dynamics](http://www.pydy.org/),(https://pypi.python.org/pypi/pydy/), [examples](#http://nbviewer.jupyter.org/github/pydy/pydy-tutorial-human-standing/tree/online-read/notebooks/)
 * [sympy classical mechanics](http://docs.sympy.org/latest/modules/physics/mechanics/index.html)

### Dynamics  
http://www.siue.edu/~rkrauss/python_intro.html  
https://www.cds.caltech.edu/~murray/wiki/Control_Systems_Library_for_Python  
http://www.vibrationdata.com/python/   
http://www.gribblelab.org/compneuro/index.html  
http://scipy.github.io/old-wiki/pages/Cookbook/CoupledSpringMassSystem  
https://ics.wofford-ecs.org/  
http://www.ni.gsu.edu/~rclewley/PyDSTool/FrontPage.html

### Aero
http://aeropy.readthedocs.io/en/latest/   
http://www.pdas.com/contents15.html  
http://www.pdas.com/aeroinfo.html  
http://www.pdas.com/aerosoft.html  
https://github.com/dgorissen/naca

### Materials
https://github.com/nagordon/ME701  
https://github.com/nagordon/mcgill_mech_530_selimb  
https://github.com/elainecraigie/MechanicsOfCompositesProject_ECraigie  
https://github.com/Rlee13/pyPLY

### Physics
https://github.com/MADEAPPS/newton-dynamics/  
https://chipmunk-physics.net/  
http://bulletphysics.org/wordpress/
http://physion.net/  
http://www.algodoo.com/  
http://box2d.org/
http://www.ode.org/  
http://vpython.org/

### Microsoft Excel
Great refernce for general python scripting. https://automatetheboringstuff.com/chapter12/  
[Use python intead of VBA with xlwings](http://xlwings.org/)  
[openpyxl](https://openpyxl.readthedocs.io/en/default/)  
[xlwings](http://xlwings.org/)  
[or just develope directly with Windows COM](http://shop.oreilly.com/product/9781565926219.do)  

### General Numeric Python
https://wiki.python.org/moin/NumericAndScientific   
[ode solver](http://hplgit.github.io/odespy/doc/web/index.html  )    
http://matplotlib.org/examples/animation/double_pendulum_animated.html  
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html  
http://www.davekuhlman.org/scipy_guide_01.html  
http://hplgit.github.io/bumpy/doc/pub/._bumpy010.html#app:resources  
https://github.com/hplgit/bumpy/blob/master/doc/src/fig-bumpy/bumpy_road_fig.py  
[Sage](http://wiki.sagemath.org/quickref  )  
http://www.gregorybard.com/SAGE.html  
http://www.people.vcu.edu/~clarson/bard-sage-for-undergraduates-2014.pdf  
http://www.sympy.org/en/features.html  
http://scipy-lectures.github.io/advanced/sympy.html  
http://docs.sympy.org/dev/tutorial/calculus.html  
http://arachnoid.com/IPython/differential_equations.html  
http://www.usna.edu/Users/math/wdj/_files/documents/teach/sm212/DiffyQ/des-book-2009-11-24.pdf   
http://www.scipy-lectures.org/advanced/sympy.html  
https://wiki.python.org/moin/NumericAndScientific 
[Cornell](http://pages.physics.cornell.edu/~sethna/StatMech/ComputerExercises/PythonSoftware/)
http://vibrationdata.com/software.htm  
http://central.scipy.org/item/84/1/simple-interactive-matplotlib-plots  
https://github.com/rougier/matplotlib-tutorial  
http://www.scipy-lectures.org/index.html  
https://github.com/rougier/numpy-100  
https://github.com/rojassergio/Learning-Scipy  
[Python numerical methods mooc](http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about)  
http://www.petercollingridge.co.uk/pygame-physics-simulation  
http://www-personal.umich.edu/~mejn/computational-physics/  
[scipy](http://docs.scipy.org/doc/scipy/reference/tutorial/)  
[sympy](http://docs.sympy.org/dev/tutorial/intro.html)  
[python fea](http://justinablack.com/pycalculix/)  
[aero python](http://lorenabarba.com/blog/announcing-aeropython/)  
[libre mechanics](http://www.libremechanics.com/)  
[pygear](http://sourceforge.net/projects/pygear/)  
[materia abaqus plugin](http://sourceforge.net/projects/materia/?source=directory)  
[pyaero](http://pyaero.sourceforge.net/)  
[kinematics](http://matplotlib.org/examples/animation/double_pendulum_animated.html)  
Pymunk   http://www.pymunk.org/en/latest/readme.html     http://chipmunk-physics.net/  
PyBullet
PyBox2D   https://github.com/pybox2d/pybox2d
PyODE     http://pyode.sourceforge.net/tutorials/tutorial2.html
http://docs.sympy.org/latest/modules/physics/mechanics/index.html
https://github.com/cdsousa/sympybotics
https://pypi.python.org/pypi/Hamilton
https://pypi.python.org/pypi/arboris
https://pypi.python.org/pypi/PyODE
https://pypi.python.org/pypi/odeViz
https://pypi.python.org/pypi/ARS
https://pypi.python.org/pypi/pymunk
http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
[Python Numerical MOOC](https://github.com/numerical-mooc/numerical-mooc) and the [course](http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about)  
https://wiki.scilab.org/Finite%20Elements%20in%20Scilab
http://pyamg.org/
http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
http://nbviewer.jupyter.org/gist/bblais/7321928
https://github.com/numpy/numpy/wiki/Numerical-software-on-Windows
http://openalea.gforge.inria.fr/doc/vplants/mechanics/doc/_build/html/user/membrane/sphere%20iso/index.html
https://github.com/pydy/scipy-2013-mechanics
http://docs.sympy.org/dev/modules/physics/mechanics/
http://download.gna.org/getfem/html/homepage/python/pygf.html
>[Sparse matrices (scipy.sparse)](http://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html)
>[Sparse Matrices in SciPy](https://scipy-lectures.github.io/advanced/scipy_sparse/)
>[The FEniCS Book: Automated Solution of Differential Equations by the Finite Element Method](http://fenicsproject.org/book/index.html#book)
>[FEniCS tutorial (Python)](http://fenicsproject.org/documentation/tutorial/)
>[SfePy: Simple Finite Elements in Python](http://sfepy.org/doc-devel/index.html)


