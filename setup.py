import setuptools

setuptools.setup(
	name = 'qmut', 
	version = "0.0.1", 
	author = "Pablo Yubero",  
	description = "This package contains a class that can handle the quantitative mutations in cobra models.",
	packages = ["cobrapy_qmut"],
	install_requires=['numpy','cobra', 'tqdm', 'datetime', 'matplotlib']  ,
	reference = "Pablo Yubero, Alvar A. Lavin, Juan F. Poyatos bioRxiv 2022.05.19.492732",
	doi = "10.1101/2022.05.19.492732"
	)
