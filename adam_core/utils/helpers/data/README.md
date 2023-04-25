# adam_core/utils/helpers/data

The csv file "sample_objects.csv" contains 27 orbits of a variety of real objects from different dynamical classes. It was generated on February 22, 2023
with the following code snippet:

```python
from adam_core.orbits.query import query_sbdb

SAMPLE_OBJECTS = {
    "Atira" : ["2020 AV2", "163693"],
    "Aten" : ["2010 TK7", "3753"],
    "Apollo" : ["54509", "2063"],
    "Amor" : ["1221", "433", "3908"],
    "Inner Main Belt" : ["434", "1876", "2001"],
    "Main Belt" : ["2", "6", "6522", "202930"],
    "Jupiter Trojans" : ["911", "1143", "1172", "3317"],
    "Centaur" : ["5145", "5335", "49036"],
    "Trans-Neptunian Objects" : ["15760", "15788", "15789"],
    "Interstellar Objects" : ["A/2017 U1"],
}
OBJECT_IDS = []
for dynamical_class in SAMPLE_OBJECTS.keys():
    OBJECT_IDS += SAMPLE_OBJECTS[dynamical_class]

orbits = query_sbdb(OBJECT_IDS)
orbits.to_df().to_csv("sample_orbits.csv", index=False)
```
