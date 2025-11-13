=================================
Mouse Cortex + Hippocampus
=================================

RNA sequencing data of single cells isolated from >20 areas of mouse cortex and hippocamopus, including ACA, AI, AUD, CA, CLA, CLA;EPd, ENTl, ENTm, GU;VISC;AIp, HIP, MOp, MOs, ORB, PAR;POST;PRE, PL;ILA, PTLp, RSP, RSPv, SSp, SSs, SSs;GU, SSs;GU;VISC, SUB;ProS, TEa;PERI;ECT, VISal;VISl;VISli, VISam;VISpm, VISp, and VISpl;VISpor.  Abbreviations match the Allen Mouse Brain Atlas.

The data set includes 1,093,785 single cells.
10xv2 sequencing reads were aligned to the mouse pre-mRNA reference transcriptome (mm10) using the 10x Genomics CellRanger pipeline (version 3.0.0) with default parameters.
For more details, please see the Documentation tab in the Cell Types web application.


Gene expression data matrix (matrix.csv)
    This file csv contains one row for every cell in the dataset and one column for every gene sequenced. The values of the matrix represent counts (UMI)
	for that gene (column) for that cell (row).

		
Medians (medians.csv)
	A table of median expression values for each gene (rows) in each cluster (columns).  Medians are calculated by first normalizing gene expression as follows: norm_data = log2(CPM(exons+introns)), and then calculating the medians independently for each gene and each cluster.
	The first row lists the cluster name (cluster_label), which matches the cell type alias shown in the Transcriptomic Explorer.
	The first column lists the unique gene identifier (gene), which in most cases is the gene symbol.


Cell metadata (metadata.csv)
* Each item of this table (except "sample_name") has three columns:
	[item]_label
		Name of the item (e.g., "V1C" would be an example under "brain_region_label")
	[item]_order
		Order that the item will be displayed on the Transcriptomics Explorer 
	[item]_color
		Color that the item will be displayed on the Transcriptomics Explorer 

* Items in the sample information table:
	sample_name
		Unique sample identifier
	cluster
		Cell type cluster name
	cell_type_accession
		Cell type accession ID (see https://portal.brain-map.org/explore/classes/nomenclature for details)
	cell_type_alias
		Cell type alias (see https://portal.brain-map.org/explore/classes/nomenclature for details).  This is the same as "cluster".
	cell_type_alt_alias
		Cell type alternative alias, if any (see https://portal.brain-map.org/explore/classes/nomenclature for details)
	cell_type_designation
		Cell type label (see https://portal.brain-map.org/explore/classes/nomenclature for details)
	class
		Broad cell class (for example, "GABAergic", "Non-neuronal", and "Glutamatergic")
	subclass
		Cell type subclass (for example, "SST", "L6 CT", and "Astrocyte")
	external_donor_name
		Unique identifier for each mouse donor
	donor_sex
		Biological sex of the donor
	cortical_layer
		Cortical layer targeted for sampling. Cells with cortical_layer=0 are non-cortical cells. 
	region
		Brain region targeted for sampling
	subregion
		Brain sub-region targeted for sampling (e.g., anterior vs. posterior), if any
	full_genotype
		Full genotype of the transgenic mouse donor
	facs_population_plan
		FACS gating criteria used to sort labeled cells
	injection_materials
		Specific virus injected into the mouse.  Blank values for this and subsequent columns indicate that no injection was performed.
	injection_method
		Method used for virus injection (Nanoject, Retro-Orbital)
	injection_roi
		Center of injection site. Abbreviations match the Allen Mouse Brain Atlas.
	propagation_type
		Type of viral propogation (retrograde, anterograde)
	
	
UMAP coordinates (tsne.csv)
UMAP coordinates (the filename is a misnomer) for each sample shown on the Transcriptomics Explorer.  UMAP is a method for dimensionality reduction of gene expression that is  well suited for data visualization.
	sample_name
		Unique sample identifier
	tsne_1
		First coordinate (again, these are actually UMAP coordinates for this dataset)
	tsne_2
		Second coordinate (again, these are actually UMAP coordinates for this dataset)

		
Taxonomy of cell types (dend.json)
	Serialized cell type hierarchy with all node information embedded in json format.
	The dendrogram shown at the top of the Transcriptomics Explorer, including the underlying cell type order, is derived from this file.
	

Taxonomy metadata (taxonomy.txt)
	Tracking taxonomy meta-data is critical for reproducibility.  This file is a draft of taxonomy meta-data to be stored.  See the "Tracking taxonomies" section at https://portal.brain-map.org/explore/classes/nomenclature for details of each descriptor.
	

Gene information (**STORED ELSEWHERE**)
* To access this file, please use the following link: http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985
* Within that zip file, the gene information is located in "mouse_VISp_2018-06-14_genes-rows.csv".  All other files can be ignored.
	gene_symbol
		Gene symbol
	gene_id
		This is an Allen Institute gene ID that can be ignored
	chromosome
		Chromosome location of gene
	gene_entrez_id
		NCBI Entrez ID
	gene_name
		Gene name

		
Gene ".gtf" file (**STORED ELSEWHERE**)
* To access this file, please use the following link: http://celltypes.brain-map.org/api/v2/well_known_file_download/502999254
.gtf is a standard format for localizing various aspects of transcripts within a specific genome and information about this format is plentiful.
As of 1 October 2019, one active link describing this format is here: https://www.gencodegenes.org/pages/data_format.html

