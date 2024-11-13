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
