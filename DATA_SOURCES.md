# Data sources

## India location lookup

`india_locations.csv` is a filtered subset of the GeoNames India gazetteer (`IN.zip`). It contains populated places with a population of at least 5,000 and administrative capitals, with coordinates and state names joined from `admin1CodesASCII.txt`.

Source: [GeoNames](https://www.geonames.org/)

GeoNames data is available under the Creative Commons Attribution 4.0 License (CC BY 4.0). The data is provided as-is. Please retain this attribution when redistributing or presenting the project.

The source extracts can be downloaded from [GeoNames export](https://www.geonames.org/export/) and the compact CSV rebuilt by running `python3 build_india_locations.py`.
