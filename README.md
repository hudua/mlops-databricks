# mlops-databricks

Here is a diagram on how to do feature-branching with Azure Databricks + Azure DevOps.

![adbado](/image/adb-ado.PNG)

To start, define a master branch folder in Databricks workspace - this is the one that only Azure DevOps pipeline would write to, and different feature branches. Here's a reference on syncing Databricks notebook to Azure DevOps repo: https://docs.microsoft.com/en-us/azure/databricks/notebooks/azure-devops-services-version-control

* You should have a notebook synced to a feature branch
* Then run the clone_to_master.py script provided here in Azure DevOps Pipeline, to automatically clone the .py script to the master branch notebook.
