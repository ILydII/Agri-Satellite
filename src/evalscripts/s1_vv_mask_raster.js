//VERSION=3
// Sentinel-1 IW GRD VV backscatter (dB) raster + dataMask, for Process API pixel-map output
// (2-band FLOAT32: [vv_db, dataMask]).
function setup() {
  return {
    input: [{ bands: ["VV", "dataMask"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [10 * Math.log10(sample.VV), sample.dataMask];
}
