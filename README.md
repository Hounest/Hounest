![Header](https://github.com/Hounest/Hounest/blob/main/IMG_20190519_130112.jpg)
# The summary of my completed and ongoing projects
My General Skills: Sociological Theory; Stata; SQL; R; Python

(1)No time to consume? The relationship between people’s time use strategy and their consumption • Master Thesis 
   
   Problem: The relationship between the leisure time of people and the consumption, whether the limitless overwork has no harm to sustainable economic growth and labor efficiency
   
   Research methods: Linear regression, Fixed effect regression, Mediation effect test
   
   Data Source: IMF, World Bank Database, Penn World Table    
  <br/> 
  <br/>
(2)The impact of Cross-straits Trade, Traders, and Media on the party identification of Taiwan people • Thesis Paper 
   
   Problem:The relationship among cross-strait trade relations, Taiwan media reports on mainland China, and political opinion in Taiwan
   
   Research methods: Linear regression, Time-series analysis
   
   Research techniques: Stata
   
   Data Source: GDELT project, Statistical yearbook of china, Election Study Center National Chengchi University   
  <br/>
  <br/>
(3)Intellectual Rent Collection: The “996” Schedule and The Formation of The “overtime consent” Myth of The Programmers • Graduation Dissertation
   
   Problem: the formation of “996” schedule: an exploitation from programmer's labor
   
   Methodology: Literature review
   
   Theory Applied: “Manufacturing Consent” of Burawoy; Theory of labor process; New-Taylorism
   
(4) Ongoing projects

   Project 1: Identify the target emerging enterprises from a given list of enterprises based on their introductions automatically
   Methods: NLP, Machine learning
   Source file: (1) Scoring.py for the demand of traditional scoring method,which can be potentially used on scoring the availability of reviews or other texts.
                (2) Machine learning part.py for the demand of automatical identifying the required reviews or other texts. 
                Tips: A proper training dataset needs to be constructed.
  
   

## Data

The two R Markdown analyses read from a `data/` directory that is not tracked
here, because the source files are far larger than is sensible to keep in a
portfolio repository. Create `data/` next to the `.Rmd` files and add:

`assignment1.Rmd` (Winter Olympics)

| File | Source |
| --- | --- |
| `data/athletes_and_events.csv` | Kaggle, *120 years of Olympic history: athletes and results* (`athlete_events.csv`, renamed) |
| `data/noc_regions.csv` | Kaggle, same dataset |
| `data/gdp_pop.csv` | World Bank GDP per capita and population, one row per NOC code |

`Assignment2_zh2448.Rmd` (NYC Airbnb)

| File | Source |
| --- | --- |
| `data/airbnb_listings.csv` | Inside Airbnb, New York City `listings.csv` |
| `data/nyc_boroughs_map/` | NYC Department of City Planning Borough Boundaries shapefile (`nybb.shp` plus its sidecar files) |
| `data/neighbourhoods.geojson` | Inside Airbnb, New York City `neighbourhoods.geojson` |

Each document checks for its inputs in the setup chunk and stops with the list
of missing paths rather than failing inside `read_csv()`.

Section 3 of `assignment1.Rmd` scrapes Wikipedia, so knitting that document
needs an internet connection.

### Package notes

Both documents were written in 2022 and have since been updated for current
CRAN packages. `Assignment2_zh2448.Rmd` now needs **tmap >= 4.0** and reads its
spatial layers with **sf** rather than the retired `rgdal`. Its basemaps use
CartoDB Positron and the NYC borough outline, so no map-tile API key is
required.
