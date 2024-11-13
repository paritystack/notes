# Github

## Quick start

- Copy the public key to the settings.
- Clone the repository using `git clone git@github.com:<username>/<repository>.git`
- Create a new branch from the main branch using `git switch -c <new branch>`
- Push the new branch to the remote repository using `git push -u origin <new branch>`


## Action Permissions

- Workflow permissions are disabled by default.
- To enable them, go to the repository settings, click on `Actions`, then click on `General`, and enable `Read and write permissions` for the `Workflow Permissions`.
- NOTE: Without workflow permissions enabled, the publishing action will fail.


## Publishing

- Select the branch `gh-pages` `/(root)` from the repository settings.
- The publishing action is defined in `.github/workflows/deploy.yml`.
- This action uses the `peaceiris/actions-mdbook` action to build and publish the book.
- The `publish_dir` is the directory that the action will publish. In this case, it is the `book` directory.
- The `github_token` is a token that the action will use to push the changes to the repository.
- The `GITHUB_TOKEN` is automatically created by Github when the repository is created.
- The `GITHUB_TOKEN` has the `contents` permission, which is required by the action.
