GUIDA ALL'UTILIZZO
Il programma di morphing in se richiede che varie condizioni vengano rispettate per quanto riguarda la disposizione delle cartelle e i nomi dei file.
Per facilitare l'utilizzo ho creato lo scipt sia in python che matlab che si occupa di creare le cartelle e rinominare i file di cui ha bisogno.
Lo script ha bisono di due parametri:
- path_image -> cartella contenente le immagini e la cartella delle maschere

E' possibile fornire allo script le immagini disposte in 3 opzioni differenti:
Option 1: immagini gia divise in fold e label
Option 2: immgini divise in fold o label (lo script deve che sono presenti immagini sotto un solo livello di sottocartelle, sara' a cura di chi lo usare decidere se considerarle fold o label)
Option 3: tutte le immagini in un unica cartella

Unico vincolo e' fornire le maschere relative alle immagini nella stessa cartella in cui si trovano quelle immagini.
Le maschere devono chiamarsi allo stesso modo della rispettiva immagine.
Per capire meglio ecco un esempio visivo:

		Option 1:			Option 2:			Option 3:
		path_image			path_image			path_image
			fold_1				fold_1				1.png
		   	   label_1			   1.png			2.png
		   		1.png		   	   2.png			3.png
		   		2.png		   	   ...				...				
		   		...		   	   mask_dir			mask_dir
		      	   	mask_dir			1.png		   	   1.png
		      		   1.png			2.png	 	   	   2.png
		      		   2.png			...			   ...
				   ...			fold_2
		    	   label_2			   1.png
		    		...		   	   2.png
		    	fold_2			   	   ...
		    	   label_1			   mask_dir				
		      		...				1.png			   
		      	   label_2				2.png	
		      		...				...		   
			...				   ...

Utilizzando lo script "data_extraction" non serve fare nulla, le immagini e le maschere vengono disposte nel giusto ordine per poter poi eseguire "renameDataset"
Una volta eseguito lo script "renameDataset" si puo' facilmente eseguire il programma di morphing senza difficolta'.


MORPHING:

Per quanto riguarda il programma di morphing il funzionamento e' il seguente:
Prendendo 2 immagini A e B, con le rispettive maschere, si creano X immagini desiderate.
Si puo decidere quante immagini creare modificando i parametri alpha.
alpha ha un range da 0 a 1, 0 l'immagine creata e' uguale all' immagine A e 1 uguale alla B.
si puo' scegliere di quanto incrementare alpha a ogni iterazione per scegliere quante immagini creare.
con alpha_min = alpha_max = 0.5 si crea una sola immagine mescolando al 50% le due immagini.

	A ---- X ---- B
	0     0.5     1  alpha

i parametri disponibili sono i seguenti:

original_path: path delle immagini originali
fourier_path: path delle maschere
output_path: path di output
alpha_min
alpha_max
alpha_increment
factor_increment = quante immagini creare rispetto alle iniziali. (es factor_increment = 10, creo 10x immagini iniziali)

il metodo che sceglie se creare o meno un immagine in base al factor_increment e' una funzione random con percentuale, non si creeranno quindi per forza il numero desiderato esatto
va considerato un possibile errore ma l'ordine di grandezza e' coerente.

funzionamento del' algoritmo di scelta:

% corrispondente alla probabilita' di scelta di un'immagine e' = immagini che si vogliono creare / immagini creabili
la funzione random quindi scegliera' in base a questa percentuale se tenere o meno le immagini
quindi per esempio se ho un dataset con 300 immagini e ne voglio creare un 10x quindi 3000 la percentuale e' 3000/ (combinazioni di 2 immagini senza ripetizioni di 300 immagini) = 3000/44850 = 6.6%
quindi solo il 6.6 % delle immagini creabili vengono effettivamente create. Dando questo dato in pasto a una funzione randomica potrebbe quindi non creare esattamente 3000 immagini ma un po di piu o un po di meno.





