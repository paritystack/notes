Object.assign(window.search, {"doc_urls":["git/cheatsheet.html#git-cheatsheet","git/cheatsheet.html#git-commands","git/cheatsheet.html#basic-setup","git/cheatsheet.html#create-a-new-branch-from-an-orphan-branch","git/github.html#github","git/github.html#quick-start","git/github.html#action-permissions","git/github.html#publishing","programming/python.html#python","programming/c.html#c","programming/cpp.html#c","programming/javascript.html#javascript","programming/bash.html#bash","programming/java.html#java","programming/rust.html#rust","programming/sql.html#sql","linux/kernel.html#kernel","linux/commands.html#linux-commands"],"index":{"documentStore":{"docInfo":{"0":{"body":0,"breadcrumbs":5,"title":2},"1":{"body":0,"breadcrumbs":5,"title":2},"10":{"body":0,"breadcrumbs":4,"title":1},"11":{"body":0,"breadcrumbs":4,"title":1},"12":{"body":0,"breadcrumbs":4,"title":1},"13":{"body":0,"breadcrumbs":4,"title":1},"14":{"body":0,"breadcrumbs":4,"title":1},"15":{"body":0,"breadcrumbs":4,"title":1},"16":{"body":0,"breadcrumbs":3,"title":1},"17":{"body":0,"breadcrumbs":5,"title":2},"2":{"body":15,"breadcrumbs":5,"title":2},"3":{"body":20,"breadcrumbs":8,"title":5},"4":{"body":0,"breadcrumbs":3,"title":1},"5":{"body":33,"breadcrumbs":4,"title":2},"6":{"body":26,"breadcrumbs":4,"title":2},"7":{"body":44,"breadcrumbs":3,"title":1},"8":{"body":0,"breadcrumbs":4,"title":1},"9":{"body":0,"breadcrumbs":4,"title":1}},"docs":{"0":{"body":"","breadcrumbs":"Git » Git Cheatsheet » Git Cheatsheet","id":"0","title":"Git Cheatsheet"},"1":{"body":"","breadcrumbs":"Git » Git Cheatsheet » Git Commands","id":"1","title":"Git Commands"},"10":{"body":"","breadcrumbs":"Programming Languages » C++ » C++","id":"10","title":"C++"},"11":{"body":"","breadcrumbs":"Programming Languages » JavaScript » JavaScript","id":"11","title":"JavaScript"},"12":{"body":"","breadcrumbs":"Programming Languages » Bash » Bash","id":"12","title":"Bash"},"13":{"body":"","breadcrumbs":"Programming Languages » Java » Java","id":"13","title":"Java"},"14":{"body":"","breadcrumbs":"Programming Languages » Rust » Rust","id":"14","title":"Rust"},"15":{"body":"","breadcrumbs":"Programming Languages » SQL » SQL","id":"15","title":"SQL"},"16":{"body":"","breadcrumbs":"Linux » Kernel » Kernel","id":"16","title":"Kernel"},"17":{"body":"","breadcrumbs":"Linux » Linux Commands » Linux Commands","id":"17","title":"Linux Commands"},"2":{"body":"git config --global user.name \"<name>\"\ngit config --global user.email \"<email>\"\ngit config --global core.editor \"vi\"","breadcrumbs":"Git » Git Cheatsheet » Basic setup","id":"2","title":"Basic setup"},"3":{"body":"git switch --orphan <new branch>\ngit commit --allow-empty -m \"Initial commit on orphan branch\"\ngit push -u origin <new branch>","breadcrumbs":"Git » Git Cheatsheet » Create a new branch from an orphan branch","id":"3","title":"Create a new branch from an orphan branch"},"4":{"body":"","breadcrumbs":"Git » Github » Github","id":"4","title":"Github"},"5":{"body":"Copy the public key to the settings. Clone the repository using git clone git@github.com:<username>/<repository>.git Create a new branch from the main branch using git switch -c <new branch> Push the new branch to the remote repository using git push -u origin <new branch>","breadcrumbs":"Git » Github » Quick start","id":"5","title":"Quick start"},"6":{"body":"Workflow permissions are disabled by default. To enable them, go to the repository settings, click on Actions, then click on General, and enable Read and write permissions for the Workflow Permissions. NOTE: Without workflow permissions enabled, the publishing action will fail.","breadcrumbs":"Git » Github » Action Permissions","id":"6","title":"Action Permissions"},"7":{"body":"Select the branch gh-pages /(root) from the repository settings. The publishing action is defined in .github/workflows/deploy.yml. This action uses the peaceiris/actions-mdbook action to build and publish the book. The publish_dir is the directory that the action will publish. In this case, it is the book directory. The github_token is a token that the action will use to push the changes to the repository. The GITHUB_TOKEN is automatically created by Github when the repository is created. The GITHUB_TOKEN has the contents permission, which is required by the action.","breadcrumbs":"Git » Github » Publishing","id":"7","title":"Publishing"},"8":{"body":"","breadcrumbs":"Programming Languages » Python » Python","id":"8","title":"Python"},"9":{"body":"","breadcrumbs":"Programming Languages » C » C","id":"9","title":"C"}},"length":18,"save":true},"fields":["title","body","breadcrumbs"],"index":{"body":{"root":{"a":{"c":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":2,"docs":{"6":{"tf":1.7320508075688772},"7":{"tf":2.449489742783178}}}}}}},"df":0,"docs":{},"l":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":1,"docs":{"3":{"tf":1.0}}}}}},"u":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}}}}}},"b":{"a":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":1,"docs":{"12":{"tf":1.0}}},"i":{"c":{"df":1,"docs":{"2":{"tf":1.0}}},"df":0,"docs":{}}}},"df":0,"docs":{},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":1,"docs":{"7":{"tf":1.4142135623730951}}}}},"r":{"a":{"df":0,"docs":{},"n":{"c":{"df":0,"docs":{},"h":{"df":3,"docs":{"3":{"tf":2.23606797749979},"5":{"tf":2.23606797749979},"7":{"tf":1.0}}}},"df":0,"docs":{}}},"df":0,"docs":{}},"u":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"d":{"df":1,"docs":{"7":{"tf":1.0}}},"df":0,"docs":{}}}}},"c":{"a":{"df":0,"docs":{},"s":{"df":0,"docs":{},"e":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":3,"docs":{"10":{"tf":1.0},"5":{"tf":1.0},"9":{"tf":1.0}},"h":{"a":{"df":0,"docs":{},"n":{"df":0,"docs":{},"g":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":0,"docs":{},"e":{"df":0,"docs":{},"e":{"df":0,"docs":{},"t":{"df":1,"docs":{"0":{"tf":1.0}}}}}}}}},"df":0,"docs":{}}},"l":{"df":0,"docs":{},"i":{"c":{"df":0,"docs":{},"k":{"df":1,"docs":{"6":{"tf":1.4142135623730951}}}},"df":0,"docs":{}},"o":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":1,"docs":{"5":{"tf":1.4142135623730951}}}}}},"o":{"df":0,"docs":{},"m":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"n":{"d":{"df":2,"docs":{"1":{"tf":1.0},"17":{"tf":1.0}}},"df":0,"docs":{}}},"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":1,"docs":{"3":{"tf":1.4142135623730951}}}}}},"n":{"df":0,"docs":{},"f":{"df":0,"docs":{},"i":{"df":0,"docs":{},"g":{"df":1,"docs":{"2":{"tf":1.7320508075688772}}}}},"t":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"p":{"df":0,"docs":{},"i":{"df":1,"docs":{"5":{"tf":1.0}}}},"r":{"df":0,"docs":{},"e":{".":{"df":0,"docs":{},"e":{"d":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":1,"docs":{"2":{"tf":1.0}}}}}}},"df":0,"docs":{}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":3,"docs":{"3":{"tf":1.0},"5":{"tf":1.0},"7":{"tf":1.4142135623730951}}}},"df":0,"docs":{}}}},"d":{"df":0,"docs":{},"e":{"df":0,"docs":{},"f":{"a":{"df":0,"docs":{},"u":{"df":0,"docs":{},"l":{"df":0,"docs":{},"t":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"i":{"df":0,"docs":{},"r":{"df":0,"docs":{},"e":{"c":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":1,"docs":{"7":{"tf":1.4142135623730951}}}}}}},"df":0,"docs":{}}},"s":{"a":{"b":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.0}}}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{},"e":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{},"p":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":1,"docs":{"3":{"tf":1.0}}}}}},"n":{"a":{"b":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.7320508075688772}}}},"df":0,"docs":{}},"df":0,"docs":{}}},"f":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.0}}}}},"df":0,"docs":{}},"g":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"h":{"df":1,"docs":{"7":{"tf":1.0}}},"i":{"df":0,"docs":{},"t":{"@":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"u":{"b":{".":{"c":{"df":0,"docs":{},"o":{"df":0,"docs":{},"m":{":":{"<":{"df":0,"docs":{},"u":{"df":0,"docs":{},"s":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"n":{"a":{"df":0,"docs":{},"m":{"df":0,"docs":{},"e":{">":{"/":{"<":{"df":0,"docs":{},"r":{"df":0,"docs":{},"e":{"df":0,"docs":{},"p":{"df":0,"docs":{},"o":{"df":0,"docs":{},"s":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"y":{">":{".":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}},"df":5,"docs":{"0":{"tf":1.0},"1":{"tf":1.0},"2":{"tf":1.7320508075688772},"3":{"tf":1.7320508075688772},"5":{"tf":1.7320508075688772}},"h":{"df":0,"docs":{},"u":{"b":{"/":{"df":0,"docs":{},"w":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"k":{"df":0,"docs":{},"f":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":0,"docs":{},"s":{"/":{"d":{"df":0,"docs":{},"e":{"df":0,"docs":{},"p":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"y":{".":{"df":0,"docs":{},"y":{"df":0,"docs":{},"m":{"df":0,"docs":{},"l":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}}}}}},"_":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.7320508075688772}}}}}}}},"df":2,"docs":{"4":{"tf":1.0},"7":{"tf":1.0}}},"df":0,"docs":{}}}}},"l":{"df":0,"docs":{},"o":{"b":{"a":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.7320508075688772}}}},"df":0,"docs":{}},"df":0,"docs":{}}},"o":{"df":1,"docs":{"6":{"tf":1.0}}}},"i":{"df":0,"docs":{},"n":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":1,"docs":{"3":{"tf":1.0}}}}}}},"j":{"a":{"df":0,"docs":{},"v":{"a":{"df":1,"docs":{"13":{"tf":1.0}},"s":{"c":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"p":{"df":0,"docs":{},"t":{"df":1,"docs":{"11":{"tf":1.0}}}}}}},"df":0,"docs":{}}},"df":0,"docs":{}}},"df":0,"docs":{}},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"l":{"df":1,"docs":{"16":{"tf":1.0}}}}}},"y":{"df":1,"docs":{"5":{"tf":1.0}}}}},"l":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":0,"docs":{},"u":{"df":0,"docs":{},"x":{"df":1,"docs":{"17":{"tf":1.0}}}}}}},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":1,"docs":{"5":{"tf":1.0}}}}},"d":{"b":{"df":0,"docs":{},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"df":0,"docs":{}},"df":1,"docs":{"3":{"tf":1.0}}},"n":{"a":{"df":0,"docs":{},"m":{"df":0,"docs":{},"e":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"df":0,"docs":{},"w":{"df":2,"docs":{"3":{"tf":1.7320508075688772},"5":{"tf":2.0}}}},"o":{"df":0,"docs":{},"t":{"df":0,"docs":{},"e":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}}}}}},"p":{"df":0,"docs":{},"h":{"a":{"df":0,"docs":{},"n":{"df":1,"docs":{"3":{"tf":1.7320508075688772}}}},"df":0,"docs":{}}}}},"p":{"a":{"df":0,"docs":{},"g":{"df":0,"docs":{},"e":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"a":{"c":{"df":0,"docs":{},"e":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"/":{"a":{"c":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{},"r":{"df":0,"docs":{},"m":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"df":0,"docs":{},"s":{"df":2,"docs":{"6":{"tf":2.23606797749979},"7":{"tf":1.0}}}}}}}},"u":{"b":{"df":0,"docs":{},"l":{"df":0,"docs":{},"i":{"c":{"df":1,"docs":{"5":{"tf":1.0}}},"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"_":{"d":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{}},"df":2,"docs":{"6":{"tf":1.0},"7":{"tf":2.0}}}}}}},"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":3,"docs":{"3":{"tf":1.0},"5":{"tf":1.4142135623730951},"7":{"tf":1.0}}}}},"y":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":1,"docs":{"8":{"tf":1.0}}}}}}}},"q":{"df":0,"docs":{},"u":{"df":0,"docs":{},"i":{"c":{"df":0,"docs":{},"k":{"df":1,"docs":{"5":{"tf":1.0}}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"e":{"a":{"d":{"df":1,"docs":{"6":{"tf":1.0}}},"df":0,"docs":{}},"df":0,"docs":{},"m":{"df":0,"docs":{},"o":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}},"p":{"df":0,"docs":{},"o":{"df":0,"docs":{},"s":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":3,"docs":{"5":{"tf":1.4142135623730951},"6":{"tf":1.0},"7":{"tf":1.7320508075688772}}}}}}}}}},"q":{"df":0,"docs":{},"u":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}}},"u":{"df":0,"docs":{},"s":{"df":0,"docs":{},"t":{"df":1,"docs":{"14":{"tf":1.0}}}}}},"s":{"df":0,"docs":{},"e":{"df":0,"docs":{},"l":{"df":0,"docs":{},"e":{"c":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}}},"t":{"df":3,"docs":{"5":{"tf":1.0},"6":{"tf":1.0},"7":{"tf":1.0}},"u":{"df":0,"docs":{},"p":{"df":1,"docs":{"2":{"tf":1.0}}}}}},"q":{"df":0,"docs":{},"l":{"df":1,"docs":{"15":{"tf":1.0}}}},"t":{"a":{"df":0,"docs":{},"r":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}},"df":0,"docs":{}},"w":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"c":{"df":0,"docs":{},"h":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}}}},"df":0,"docs":{}}}}},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"u":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}},"s":{"df":2,"docs":{"5":{"tf":1.7320508075688772},"7":{"tf":1.4142135623730951}},"e":{"df":0,"docs":{},"r":{".":{"df":0,"docs":{},"e":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{}}},"n":{"a":{"df":0,"docs":{},"m":{"df":1,"docs":{"2":{"tf":1.0}}}},"df":0,"docs":{}}},"df":0,"docs":{}}}}},"v":{"df":0,"docs":{},"i":{"df":1,"docs":{"2":{"tf":1.0}}}},"w":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"o":{"df":0,"docs":{},"u":{"df":0,"docs":{},"t":{"df":1,"docs":{"6":{"tf":1.0}}}}}}}},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"k":{"df":0,"docs":{},"f":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":1,"docs":{"6":{"tf":1.7320508075688772}}}}}}}}},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"e":{"df":1,"docs":{"6":{"tf":1.0}}}}}}}}},"breadcrumbs":{"root":{"a":{"c":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":2,"docs":{"6":{"tf":2.0},"7":{"tf":2.449489742783178}}}}}}},"df":0,"docs":{},"l":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":1,"docs":{"3":{"tf":1.0}}}}}},"u":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}}}}}},"b":{"a":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":1,"docs":{"12":{"tf":1.7320508075688772}}},"i":{"c":{"df":1,"docs":{"2":{"tf":1.4142135623730951}}},"df":0,"docs":{}}}},"df":0,"docs":{},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":1,"docs":{"7":{"tf":1.4142135623730951}}}}},"r":{"a":{"df":0,"docs":{},"n":{"c":{"df":0,"docs":{},"h":{"df":3,"docs":{"3":{"tf":2.6457513110645907},"5":{"tf":2.23606797749979},"7":{"tf":1.0}}}},"df":0,"docs":{}}},"df":0,"docs":{}},"u":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"d":{"df":1,"docs":{"7":{"tf":1.0}}},"df":0,"docs":{}}}}},"c":{"a":{"df":0,"docs":{},"s":{"df":0,"docs":{},"e":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":3,"docs":{"10":{"tf":1.7320508075688772},"5":{"tf":1.0},"9":{"tf":1.7320508075688772}},"h":{"a":{"df":0,"docs":{},"n":{"df":0,"docs":{},"g":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":0,"docs":{},"e":{"df":0,"docs":{},"e":{"df":0,"docs":{},"t":{"df":4,"docs":{"0":{"tf":1.7320508075688772},"1":{"tf":1.0},"2":{"tf":1.0},"3":{"tf":1.0}}}}}}}}},"df":0,"docs":{}}},"l":{"df":0,"docs":{},"i":{"c":{"df":0,"docs":{},"k":{"df":1,"docs":{"6":{"tf":1.4142135623730951}}}},"df":0,"docs":{}},"o":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":1,"docs":{"5":{"tf":1.4142135623730951}}}}}},"o":{"df":0,"docs":{},"m":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"n":{"d":{"df":2,"docs":{"1":{"tf":1.4142135623730951},"17":{"tf":1.7320508075688772}}},"df":0,"docs":{}}},"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":1,"docs":{"3":{"tf":1.4142135623730951}}}}}},"n":{"df":0,"docs":{},"f":{"df":0,"docs":{},"i":{"df":0,"docs":{},"g":{"df":1,"docs":{"2":{"tf":1.7320508075688772}}}}},"t":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"p":{"df":0,"docs":{},"i":{"df":1,"docs":{"5":{"tf":1.0}}}},"r":{"df":0,"docs":{},"e":{".":{"df":0,"docs":{},"e":{"d":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":1,"docs":{"2":{"tf":1.0}}}}}}},"df":0,"docs":{}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":3,"docs":{"3":{"tf":1.4142135623730951},"5":{"tf":1.0},"7":{"tf":1.4142135623730951}}}},"df":0,"docs":{}}}},"d":{"df":0,"docs":{},"e":{"df":0,"docs":{},"f":{"a":{"df":0,"docs":{},"u":{"df":0,"docs":{},"l":{"df":0,"docs":{},"t":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"i":{"df":0,"docs":{},"r":{"df":0,"docs":{},"e":{"c":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":1,"docs":{"7":{"tf":1.4142135623730951}}}}}}},"df":0,"docs":{}}},"s":{"a":{"b":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.0}}}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{},"e":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{},"p":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":1,"docs":{"3":{"tf":1.0}}}}}},"n":{"a":{"b":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.7320508075688772}}}},"df":0,"docs":{}},"df":0,"docs":{}}},"f":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"6":{"tf":1.0}}}}},"df":0,"docs":{}},"g":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"h":{"df":1,"docs":{"7":{"tf":1.0}}},"i":{"df":0,"docs":{},"t":{"@":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"u":{"b":{".":{"c":{"df":0,"docs":{},"o":{"df":0,"docs":{},"m":{":":{"<":{"df":0,"docs":{},"u":{"df":0,"docs":{},"s":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"n":{"a":{"df":0,"docs":{},"m":{"df":0,"docs":{},"e":{">":{"/":{"<":{"df":0,"docs":{},"r":{"df":0,"docs":{},"e":{"df":0,"docs":{},"p":{"df":0,"docs":{},"o":{"df":0,"docs":{},"s":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"y":{">":{".":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}},"df":8,"docs":{"0":{"tf":2.0},"1":{"tf":2.0},"2":{"tf":2.23606797749979},"3":{"tf":2.23606797749979},"4":{"tf":1.0},"5":{"tf":2.0},"6":{"tf":1.0},"7":{"tf":1.0}},"h":{"df":0,"docs":{},"u":{"b":{"/":{"df":0,"docs":{},"w":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"k":{"df":0,"docs":{},"f":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":0,"docs":{},"s":{"/":{"d":{"df":0,"docs":{},"e":{"df":0,"docs":{},"p":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"y":{".":{"df":0,"docs":{},"y":{"df":0,"docs":{},"m":{"df":0,"docs":{},"l":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}}}}}},"_":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.7320508075688772}}}}}}}},"df":4,"docs":{"4":{"tf":1.7320508075688772},"5":{"tf":1.0},"6":{"tf":1.0},"7":{"tf":1.4142135623730951}}},"df":0,"docs":{}}}}},"l":{"df":0,"docs":{},"o":{"b":{"a":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.7320508075688772}}}},"df":0,"docs":{}},"df":0,"docs":{}}},"o":{"df":1,"docs":{"6":{"tf":1.0}}}},"i":{"df":0,"docs":{},"n":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":1,"docs":{"3":{"tf":1.0}}}}}}},"j":{"a":{"df":0,"docs":{},"v":{"a":{"df":1,"docs":{"13":{"tf":1.7320508075688772}},"s":{"c":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"p":{"df":0,"docs":{},"t":{"df":1,"docs":{"11":{"tf":1.7320508075688772}}}}}}},"df":0,"docs":{}}},"df":0,"docs":{}}},"df":0,"docs":{}},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"l":{"df":1,"docs":{"16":{"tf":1.7320508075688772}}}}}},"y":{"df":1,"docs":{"5":{"tf":1.0}}}}},"l":{"a":{"df":0,"docs":{},"n":{"df":0,"docs":{},"g":{"df":0,"docs":{},"u":{"a":{"df":0,"docs":{},"g":{"df":8,"docs":{"10":{"tf":1.0},"11":{"tf":1.0},"12":{"tf":1.0},"13":{"tf":1.0},"14":{"tf":1.0},"15":{"tf":1.0},"8":{"tf":1.0},"9":{"tf":1.0}}}},"df":0,"docs":{}}}}},"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":0,"docs":{},"u":{"df":0,"docs":{},"x":{"df":2,"docs":{"16":{"tf":1.0},"17":{"tf":2.0}}}}}}},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":1,"docs":{"5":{"tf":1.0}}}}},"d":{"b":{"df":0,"docs":{},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":1,"docs":{"7":{"tf":1.0}}}}}},"df":0,"docs":{}},"df":1,"docs":{"3":{"tf":1.0}}},"n":{"a":{"df":0,"docs":{},"m":{"df":0,"docs":{},"e":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"df":0,"docs":{},"w":{"df":2,"docs":{"3":{"tf":2.0},"5":{"tf":2.0}}}},"o":{"df":0,"docs":{},"t":{"df":0,"docs":{},"e":{"df":1,"docs":{"6":{"tf":1.0}}}}}},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}}}}}},"p":{"df":0,"docs":{},"h":{"a":{"df":0,"docs":{},"n":{"df":1,"docs":{"3":{"tf":2.0}}}},"df":0,"docs":{}}}}},"p":{"a":{"df":0,"docs":{},"g":{"df":0,"docs":{},"e":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{},"e":{"a":{"c":{"df":0,"docs":{},"e":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"/":{"a":{"c":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}},"df":0,"docs":{}},"df":0,"docs":{}}}}}}},"df":0,"docs":{}},"df":0,"docs":{},"r":{"df":0,"docs":{},"m":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"df":0,"docs":{},"s":{"df":2,"docs":{"6":{"tf":2.449489742783178},"7":{"tf":1.0}}}}}}}},"r":{"df":0,"docs":{},"o":{"df":0,"docs":{},"g":{"df":0,"docs":{},"r":{"a":{"df":0,"docs":{},"m":{"df":8,"docs":{"10":{"tf":1.0},"11":{"tf":1.0},"12":{"tf":1.0},"13":{"tf":1.0},"14":{"tf":1.0},"15":{"tf":1.0},"8":{"tf":1.0},"9":{"tf":1.0}}}},"df":0,"docs":{}}}}},"u":{"b":{"df":0,"docs":{},"l":{"df":0,"docs":{},"i":{"c":{"df":1,"docs":{"5":{"tf":1.0}}},"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"_":{"d":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":1,"docs":{"7":{"tf":1.0}}}}},"df":0,"docs":{}},"df":2,"docs":{"6":{"tf":1.0},"7":{"tf":2.23606797749979}}}}}}},"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":3,"docs":{"3":{"tf":1.0},"5":{"tf":1.4142135623730951},"7":{"tf":1.0}}}}},"y":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":1,"docs":{"8":{"tf":1.7320508075688772}}}}}}}},"q":{"df":0,"docs":{},"u":{"df":0,"docs":{},"i":{"c":{"df":0,"docs":{},"k":{"df":1,"docs":{"5":{"tf":1.4142135623730951}}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"e":{"a":{"d":{"df":1,"docs":{"6":{"tf":1.0}}},"df":0,"docs":{}},"df":0,"docs":{},"m":{"df":0,"docs":{},"o":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}},"p":{"df":0,"docs":{},"o":{"df":0,"docs":{},"s":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":3,"docs":{"5":{"tf":1.4142135623730951},"6":{"tf":1.0},"7":{"tf":1.7320508075688772}}}}}}}}}},"q":{"df":0,"docs":{},"u":{"df":0,"docs":{},"i":{"df":0,"docs":{},"r":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"o":{"df":0,"docs":{},"o":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}}},"u":{"df":0,"docs":{},"s":{"df":0,"docs":{},"t":{"df":1,"docs":{"14":{"tf":1.7320508075688772}}}}}},"s":{"df":0,"docs":{},"e":{"df":0,"docs":{},"l":{"df":0,"docs":{},"e":{"c":{"df":0,"docs":{},"t":{"df":1,"docs":{"7":{"tf":1.0}}}},"df":0,"docs":{}}},"t":{"df":3,"docs":{"5":{"tf":1.0},"6":{"tf":1.0},"7":{"tf":1.0}},"u":{"df":0,"docs":{},"p":{"df":1,"docs":{"2":{"tf":1.4142135623730951}}}}}},"q":{"df":0,"docs":{},"l":{"df":1,"docs":{"15":{"tf":1.7320508075688772}}}},"t":{"a":{"df":0,"docs":{},"r":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.4142135623730951}}}}},"df":0,"docs":{}},"w":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"c":{"df":0,"docs":{},"h":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}}}},"df":0,"docs":{}}}}},"t":{"df":0,"docs":{},"o":{"df":0,"docs":{},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"n":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"u":{"df":2,"docs":{"3":{"tf":1.0},"5":{"tf":1.0}},"s":{"df":2,"docs":{"5":{"tf":1.7320508075688772},"7":{"tf":1.4142135623730951}},"e":{"df":0,"docs":{},"r":{".":{"df":0,"docs":{},"e":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"i":{"df":0,"docs":{},"l":{"df":1,"docs":{"2":{"tf":1.0}}}}},"df":0,"docs":{}}},"n":{"a":{"df":0,"docs":{},"m":{"df":1,"docs":{"2":{"tf":1.0}}}},"df":0,"docs":{}}},"df":0,"docs":{}}}}},"v":{"df":0,"docs":{},"i":{"df":1,"docs":{"2":{"tf":1.0}}}},"w":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"o":{"df":0,"docs":{},"u":{"df":0,"docs":{},"t":{"df":1,"docs":{"6":{"tf":1.0}}}}}}}},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"k":{"df":0,"docs":{},"f":{"df":0,"docs":{},"l":{"df":0,"docs":{},"o":{"df":0,"docs":{},"w":{"df":1,"docs":{"6":{"tf":1.7320508075688772}}}}}}}}},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":0,"docs":{},"e":{"df":1,"docs":{"6":{"tf":1.0}}}}}}}}},"title":{"root":{"a":{"c":{"df":0,"docs":{},"t":{"df":0,"docs":{},"i":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":1,"docs":{"6":{"tf":1.0}}}}}}},"df":0,"docs":{}},"b":{"a":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":1,"docs":{"12":{"tf":1.0}}},"i":{"c":{"df":1,"docs":{"2":{"tf":1.0}}},"df":0,"docs":{}}}},"df":0,"docs":{},"r":{"a":{"df":0,"docs":{},"n":{"c":{"df":0,"docs":{},"h":{"df":1,"docs":{"3":{"tf":1.4142135623730951}}}},"df":0,"docs":{}}},"df":0,"docs":{}}},"c":{"df":2,"docs":{"10":{"tf":1.0},"9":{"tf":1.0}},"h":{"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":0,"docs":{},"e":{"df":0,"docs":{},"e":{"df":0,"docs":{},"t":{"df":1,"docs":{"0":{"tf":1.0}}}}}}}}},"df":0,"docs":{}}},"o":{"df":0,"docs":{},"m":{"df":0,"docs":{},"m":{"a":{"df":0,"docs":{},"n":{"d":{"df":2,"docs":{"1":{"tf":1.0},"17":{"tf":1.0}}},"df":0,"docs":{}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"e":{"a":{"df":0,"docs":{},"t":{"df":1,"docs":{"3":{"tf":1.0}}}},"df":0,"docs":{}}}},"df":0,"docs":{},"g":{"df":0,"docs":{},"i":{"df":0,"docs":{},"t":{"df":2,"docs":{"0":{"tf":1.0},"1":{"tf":1.0}},"h":{"df":0,"docs":{},"u":{"b":{"df":1,"docs":{"4":{"tf":1.0}}},"df":0,"docs":{}}}}}},"j":{"a":{"df":0,"docs":{},"v":{"a":{"df":1,"docs":{"13":{"tf":1.0}},"s":{"c":{"df":0,"docs":{},"r":{"df":0,"docs":{},"i":{"df":0,"docs":{},"p":{"df":0,"docs":{},"t":{"df":1,"docs":{"11":{"tf":1.0}}}}}}},"df":0,"docs":{}}},"df":0,"docs":{}}},"df":0,"docs":{}},"k":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"l":{"df":1,"docs":{"16":{"tf":1.0}}}}}}}},"l":{"df":0,"docs":{},"i":{"df":0,"docs":{},"n":{"df":0,"docs":{},"u":{"df":0,"docs":{},"x":{"df":1,"docs":{"17":{"tf":1.0}}}}}}},"n":{"df":0,"docs":{},"e":{"df":0,"docs":{},"w":{"df":1,"docs":{"3":{"tf":1.0}}}}},"o":{"df":0,"docs":{},"r":{"df":0,"docs":{},"p":{"df":0,"docs":{},"h":{"a":{"df":0,"docs":{},"n":{"df":1,"docs":{"3":{"tf":1.0}}}},"df":0,"docs":{}}}}},"p":{"df":0,"docs":{},"e":{"df":0,"docs":{},"r":{"df":0,"docs":{},"m":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"df":0,"docs":{},"s":{"df":1,"docs":{"6":{"tf":1.0}}}}}}}},"u":{"b":{"df":0,"docs":{},"l":{"df":0,"docs":{},"i":{"df":0,"docs":{},"s":{"df":0,"docs":{},"h":{"df":1,"docs":{"7":{"tf":1.0}}}}}}},"df":0,"docs":{}},"y":{"df":0,"docs":{},"t":{"df":0,"docs":{},"h":{"df":0,"docs":{},"o":{"df":0,"docs":{},"n":{"df":1,"docs":{"8":{"tf":1.0}}}}}}}},"q":{"df":0,"docs":{},"u":{"df":0,"docs":{},"i":{"c":{"df":0,"docs":{},"k":{"df":1,"docs":{"5":{"tf":1.0}}}},"df":0,"docs":{}}}},"r":{"df":0,"docs":{},"u":{"df":0,"docs":{},"s":{"df":0,"docs":{},"t":{"df":1,"docs":{"14":{"tf":1.0}}}}}},"s":{"df":0,"docs":{},"e":{"df":0,"docs":{},"t":{"df":0,"docs":{},"u":{"df":0,"docs":{},"p":{"df":1,"docs":{"2":{"tf":1.0}}}}}},"q":{"df":0,"docs":{},"l":{"df":1,"docs":{"15":{"tf":1.0}}}},"t":{"a":{"df":0,"docs":{},"r":{"df":0,"docs":{},"t":{"df":1,"docs":{"5":{"tf":1.0}}}}},"df":0,"docs":{}}}}}},"lang":"English","pipeline":["trimmer","stopWordFilter","stemmer"],"ref":"id","version":"0.9.5"},"results_options":{"limit_results":30,"teaser_word_count":30},"search_options":{"bool":"OR","expand":true,"fields":{"body":{"boost":1},"breadcrumbs":{"boost":1},"title":{"boost":2}}}});