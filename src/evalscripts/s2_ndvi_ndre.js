//VERSION=3
// Sentinel-2 L2A: NDVI + NDRE, cloud/shadow/cirrus masked out via Scene Classification (SCL).
// SCL codes excluded: 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high prob, 10 thin cirrus.
function setup() {
  return {
    input: [{ bands: ["B04", "B05", "B08", "SCL", "dataMask"], units: "DN" }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "ndre", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(sample) {
  var badScl = [0, 1, 3, 8, 9, 10];
  var valid = sample.dataMask;
  if (badScl.indexOf(sample.SCL) !== -1) {
    valid = 0;
  }
  var ndvi = index(sample.B08, sample.B04);
  var ndre = index(sample.B08, sample.B05);
  return {
    ndvi: [ndvi],
    ndre: [ndre],
    dataMask: [valid]
  };
}
