//VERSION=3
// Sentinel-1 IW GRD, orthorectified: VV and VH backscatter in dB.
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: [
      { id: "vv_db", bands: 1, sampleType: "FLOAT32" },
      { id: "vh_db", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function toDb(linear) {
  return 10 * Math.log10(linear);
}

function evaluatePixel(sample) {
  return {
    vv_db: [toDb(sample.VV)],
    vh_db: [toDb(sample.VH)],
    dataMask: [sample.dataMask]
  };
}
