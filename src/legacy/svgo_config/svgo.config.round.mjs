export default {
  plugins: [
    {
      name: 'preset-default',
      params: {
        overrides: {
          // disable a default plugin
        //   cleanupIds: false,
          // customize the params of a default plugin
        //   inlineStyles: {
        //     onlyMatchedOnce: false,
        //   },
            mergePaths: false,
            convertPathData: {
              applyTransforms: false,
              applyTransformsStroked: false,
              straightCurves: false,
              convertToQ: false,
              lineShorthands: false,
              convertToZ: false,
              curveSmoothShorthands: false,
              floatPrecision: 0,
              transformPrecision: 10,
              smartArcRounding: false,
              removeUseless: false,
              collapseRepeated: false,
              utilizeAbsolute: false,
              negativeExtraSpace: false,
              forceAbsolutePath: false
            },
            // convertPathDatao: false,
            cleanupNumericValues: {
              floatPrecision:0,
            },
            convertTransform: {
              convertToShorts: false,
              floatPrecision: 0,
              transformPrecision: 10,
              matrixToTransform: false,
              shortTranslate: false,
              shortScale: false,
              shortRotate: false,
              removeUseless: false,
              collapseIntoOne: false
            }
        },
      },
    },
  ],
};
