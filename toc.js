// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="index.html">Introduction</a></li><li class="chapter-item expanded "><a href="git/index.html"><strong aria-hidden="true">1.</strong> Git</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="git/cheatsheet.html"><strong aria-hidden="true">1.1.</strong> Git Cheatsheet</a></li><li class="chapter-item expanded "><a href="git/commands.html"><strong aria-hidden="true">1.2.</strong> Git Commands</a></li><li class="chapter-item expanded "><a href="git/github.html"><strong aria-hidden="true">1.3.</strong> Github</a></li></ol></li><li class="chapter-item expanded "><a href="programming/index.html"><strong aria-hidden="true">2.</strong> Programming Languages</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="programming/python.html"><strong aria-hidden="true">2.1.</strong> Python</a></li><li class="chapter-item expanded "><a href="programming/c.html"><strong aria-hidden="true">2.2.</strong> C</a></li><li class="chapter-item expanded "><a href="programming/cpp.html"><strong aria-hidden="true">2.3.</strong> C++</a></li><li class="chapter-item expanded "><a href="programming/javascript.html"><strong aria-hidden="true">2.4.</strong> JavaScript</a></li><li class="chapter-item expanded "><a href="programming/bash.html"><strong aria-hidden="true">2.5.</strong> Bash</a></li><li class="chapter-item expanded "><a href="programming/java.html"><strong aria-hidden="true">2.6.</strong> Java</a></li><li class="chapter-item expanded "><a href="programming/rust.html"><strong aria-hidden="true">2.7.</strong> Rust</a></li><li class="chapter-item expanded "><a href="programming/sql.html"><strong aria-hidden="true">2.8.</strong> SQL</a></li></ol></li><li class="chapter-item expanded "><a href="linux/index.html"><strong aria-hidden="true">3.</strong> Linux</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="linux/kernel.html"><strong aria-hidden="true">3.1.</strong> Kernel</a></li><li class="chapter-item expanded "><a href="linux/commands.html"><strong aria-hidden="true">3.2.</strong> Linux Commands</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
