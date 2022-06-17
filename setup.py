import setuptools

setuptools.setup(
	name = 'qmut', 
	version = "0.0.1", 
	author = "Pablo Yubero",  
	description = "This package contains a class that can handle quantitative mutations in cobra models. More info available at 'Pablo Yubero, Alvar A. Lavin, Juan F. Poyatos bioRxiv 2022.05.19.492732' ",
	packages = ["cobrapy_qmut"],
	install_requires=['numpy','cobra', 'tqdm', 'datetime', 'matplotlib']  ,
	reference = "Pablo Yubero, Alvar A. Lavin, Juan F. Poyatos bioRxiv 2022.05.19.492732",
	doi = "10.1101/2022.05.19.492732"
	)
