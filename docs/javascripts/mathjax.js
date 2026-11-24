window.MathJax = {
  loader: { load: ["[tex]/mathtools", "[tex]/boldsymbol", "[tex]/cancel"] },
  tex: {
    packages: { "[+]": ["mathtools", "boldsymbol", "cancel"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "none",
    macros: {
      R: "\\mathbb{R}",
      E: "\\mathbb{E}",
      softmax: "\\operatorname{softmax}",
      argmax: "\\operatorname*{arg\\,max}",
      argmin: "\\operatorname*{arg\\,min}",
      KL: "D_{\\mathrm{KL}}",
      diag: "\\operatorname{diag}",
      tr: "\\operatorname{tr}",
      rank: "\\operatorname{rank}",
      Var: "\\operatorname{Var}",
      Cov: "\\operatorname{Cov}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  chtml: { scale: 1.02 }
};

document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
