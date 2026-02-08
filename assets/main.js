document.documentElement.classList.add("js");

const reveals = document.querySelectorAll(".reveal");
const observer = new IntersectionObserver(
  (entries, obs) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }
      entry.target.classList.add("visible");
      obs.unobserve(entry.target);
    });
  },
  {
    threshold: 0.16,
    rootMargin: "0px 0px -28px 0px"
  }
);

reveals.forEach((el) => observer.observe(el));

const copyBtn = document.querySelector(".copy-btn");
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const targetId = copyBtn.getAttribute("data-copy-target");
    const block = targetId ? document.getElementById(targetId) : null;
    if (!block) {
      return;
    }
    try {
      await navigator.clipboard.writeText(block.textContent || "");
      const original = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = original;
      }, 1200);
    } catch (_err) {
      copyBtn.textContent = "Copy failed";
    }
  });
}
