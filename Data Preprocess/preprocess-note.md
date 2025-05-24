# Data Processing Documentation

This document details the processing steps applied to the `accident.csv` and `vehicle.csv` datasets to generate derived categorical columns for analysis.

## 1. Processing `accident.csv`

The `accident.csv` dataset was processed to create new columns related to light conditions and road geometry.

### 1.1 Light Condition Processing

#### Original Data
The original `LIGHT_CONDITION` column contained numeric values (1-9) that represent different lighting conditions:

| Code | Description |
|------|-------------|
| 1 | Day |
| 2 | Dusk/Dawn |
| 3 | Dark street lights on |
| 4 | Dark street lights off |
| 5 | Dark no street lights |
| 6 | Dark street lights unknown |
| 9 | Unknown |

Here is the link for meaning of each value: https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data/resource/6d16124d-1f59-478a-baf8-a139dc5742dc 

#### Value Counts
```
LIGHT_CONDITION
1    119343
3     27332
2     15407
5      9449
9      4772
6      2005
4       387
```

#### Derived Column

**`CATEGORIZED_LIGHT_CONDITION`**: Groups the original values into broader categories for simplified analysis.

| Original Value | Categorized Value        |
|----------------|-------------------------|
| 1 (Day)        | "Daylight"              |
| 2 (Dusk/Dawn)  | "Limited Light"         |
| 3 (Dark street lights on) | "Dark with Lighting" |
| 4 (Dark street lights off) | "Dark without Lighting" |
| 5 (Dark no street lights)  | "Dark without Lighting" |
| 6 (Dark street lights unknown) | "Unknown" |
| 9 (Unknown)    | "Unknown"               |

#### Treatment of Unknown Values

- Values 6 (Dark street lights unknown) and 9 (Unknown) were intentionally mapped to "Unknown" in the `CATEGORIZED_LIGHT_CONDITION` column.
- This decision was made because these values represent about 3.8% of the dataset (2,005 + 4,772 = 6,777 out of 178,695 records), and including them as a separate category helps avoid bias.

### 1.2 Road Geometry Processing

#### Original Data
The original `ROAD_GEOMETRY_DESC` column contained string values describing different road configurations:

#### Value Counts
```
ROAD_GEOMETRY_DESC
Not at intersection      92444
Cross intersection       41039
T intersection           40656
Multiple intersection     3486
Y intersection             610
Unknown                    271
Dead end                   172
Private property            11
Road closure                 6
```

#### Derived Column

**`CATEGORIZED_ROAD_GEOMETRY`**: Groups the original values into broader categories for simplified analysis.

| Original Value           | Categorized Value         |
|--------------------------|--------------------------|
| "Not at intersection"    | "Not at intersection"    |
| "Cross intersection"     | "Intersection"           |
| "T intersection"         | "Intersection"           |
| "Multiple intersection"  | "Intersection"           |
| "Y intersection"         | "Intersection"           |
| "Dead end"               | "Special Road Feature"   |
| "Road closure"           | "Special Road Feature"   |
| "Unknown"                | "Other/Unknown"          |
| "Private property"       | "Other/Unknown"          |

#### Treatment of Unknown Values

- Values "Unknown" and "Private property" were intentionally mapped to "Other/Unknown" in the `CATEGORIZED_ROAD_GEOMETRY` column.
- This decision was made because these values represent only 0.16% of the dataset (271 + 11 = 282 out of 178,695 records).

## 2. Processing `vehicle.csv`

The `vehicle.csv` dataset was processed to create new columns related to road surface conditions.

### 2.1 Road Surface Processing

#### Original Data
The original `ROAD_SURFACE_TYPE_DESC` column contained string values describing different road surface types:

#### Value Counts
```
ROAD_SURFACE_TYPE_DESC
Paved        309360
Gravel        11859
Not known      2733
Unpaved        1936
NaN               1
```

#### Derived Column

**`CATEGORIZED_ROAD_SURFACE`**: Groups the original values into broader categories for simplified analysis.

| Original Value | Categorized Value |
|----------------|------------------|
| "Paved"        | "Paved"          |
| "Gravel"       | "Gravel"         |
| "Unpaved"      | "Unpaved"        |
| "Not known"    | "Unknown"        |

#### Treatment of Unknown and NaN Values

- The 1 row with NaN in the original `ROAD_SURFACE_TYPE_DESC` column was removed from the dataset.
- Values "Not known" were intentionally mapped to "Unknown" in the `CATEGORIZED_ROAD_SURFACE` column.
- This decision was made because "Not known" values represent less than 1% of the dataset (2,733 out of 325,889 records).

## 3. Output Files

Two new CSV files were generated:

1. **`updated_accident.csv`** containing the following columns:
   - `ROAD_GEOMETRY_DESC` (original)
   - `ROAD_GEOMETRY` (original encoded column of the original columns)
   - `LIGHT_CONDITION` (original)
   - `SEVERITY` (original)
   - `CATEGORIZED_ROAD_GEOMETRY` (derived)
   - `CATEGORIZED_LIGHT_CONDITION` (derived)

2. **`updated_vehicle.csv`** containing the following columns:
   - `ACCIDENT_NO` (original - for joining with other datasets)
   - `ROAD_SURFACE_TYPE` - the original encoded column of ROAD_SURFACE_TYPE_DESC
   - `ROAD_SURFACE_TYPE_DESC` (original)
   - `CATEGORIZED_ROAD_SURFACE` (derived)

## 4. Notes for Analysis

- The categorized columns can be used for visualization and descriptive statistics.