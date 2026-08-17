//VERSION=3
// Sentinel-2 L2A NDVI raster + cloud/shadow/cirrus mask, for Process API pixel-map output
// (2-band FLOAT32: [ndvi, dataMask]). Same SCL exclusion list and single-least-cloudy-scene
// mosaicking as s2_ndvi_ndre.js (mosaickingOrder=leastCC is set on the request's input data).
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"], units: "DN" }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  var badScl = [0, 1, 3, 8, 9, 10];
  var valid = sample.dataMask;
  if (badScl.indexOf(sample.SCL) !== -1) valid = 0;
  var ndvi = index(sample.B08, sample.B04);
  return [ndvi, valid];
}
