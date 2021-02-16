# mlops-databricks

Here is a diagram on how to do feature-branching with Azure Databricks + Azure DevOps.

![adbado](/image/adb-ado2.PNG)

To start, define a master branch folder in Databricks workspace - this is the one that only Azure DevOps pipeline would write to, and different feature branches. Here's a reference on syncing Databricks notebook to Azure DevOps repo: https://docs.microsoft.com/en-us/azure/databricks/notebooks/azure-devops-services-version-control

* You should have a notebook synced to a feature branch, so the notebook .py script will be present in the feature branch
* Then run the clone_to_master.py script provided here in Azure DevOps Pipeline, to automatically clone the .py script to the master branch notebook.
* Once you approve and complete a pull request merging the feature branch into master, the notebook .py script in master will be cloned to the master notebook in Azure Databricks

Please note this is done at this time at a per notebook level. If you have multiple notebooks, you will need some orchestrating logic in the Pipeline to clone multiple notebooks.

For creating a new feature branch, create the feature branch in Azure DevOps, clone the notebook from master folder in Databricks, and sync it to the newly created branch.
