This Document Describes the Data used in Training the Frosty Dawgs SnowML Model. <br>
Sections include  
  - Raw Data
  - Processing Pipeline
  - Model Ready Data

  If you are most interested in understanding the final data used in the model, jump straight to Model Ready Data!


# Model Ready Data <br>
Data ready to be deployed into the Frosty Dawgs SnowML model can be downloaded from the S3 bucket named "dawgs-model-ready". <br>
Each file contains data for one discrete HUC unit (watershed/subwatershed) 

**Naming convention**
- Each file is named "model_ready_{huc_id}.csv"
- For example, "model_ready_huc170200090302.csv" contains data for huc_id 170200090302, a HUC 12 (sub-watershed) huc unit that is part of the Chelan basin.

**Time Period** <br>
Each file contains data for the water years 1984 through 2022. <br>
Each variable is measured daily for the period "1983-10-01" through "2022-09-30"
