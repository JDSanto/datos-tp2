generarInforme:
	pdflatex -output-directory=informe informe/caratula.tex
	pandoc --toc --toc-depth=5 informe/informe.md -o informe/informe-cuerpo.pdf
	pdfunite informe/caratula.pdf informe/informe-cuerpo.pdf informe/informe.pdf
