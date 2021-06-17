## Protocol for project workflow  (Subject to change)
   
* MASTER branch should be Deployable (It’s supposed to be deployed and built by anyone at any time, without errors!)

* Tracking branch is created from master, the develop branch (This branch will contain the complete history of the project)

* All features will be worked on from their own respective working branch, which is branched from "develop"
    * Upon a particular feature being completed, we merge back into the develop branch using rebase
    * Then, as good practice, we delete our feature branch before merging develop to master

### IN CASE OF HOTFIXES
* Branch from master as "hotfix", fix the issue, merge back into master immediately