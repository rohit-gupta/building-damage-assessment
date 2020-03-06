
import shapely.wkt
from shapely.geometry import mapping


wkt_str = "POLYGON ((800.58672364816 177.3228698828206, 818.8664968408225 177.3228698828206, 820.085148387 223.2254114555063, 797.3369861916867 224.4440630016838, 795.3059002813909 202.914552352548, 792.8685971890359 195.602643075483, 799.3680721019825 186.2596478881222, 800.58672364816 177.3228698828206))"
x = list(mapping(shapely.wkt.loads(wkt_str))['coordinates'][0])
print(x)
