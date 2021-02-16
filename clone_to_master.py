import sys
import base64
from databricks_api import DatabricksAPI

token = sys.argv[1]
notebook_path = sys.argv[2]
notebook_name = sys.argv[3]
print(notebook_name)

db = DatabricksAPI(
    host="https://<region>.azuredatabricks.net",
    token=token
)

with open("notebooks/{}".format(notebook_path)) as file:
    data = file.read()

pipeline_main_file_name = "/ProjectName/master/{}-auto-added".format(notebook_name.replace(".py", ""))

encodedBytes = base64.b64encode(data.encode("utf-8"))
encodedStr = str(encodedBytes, "utf-8")

db.workspace.import_workspace(
    pipeline_main_file_name,
    format="SOURCE",
    language="PYTHON",
    content=encodedStr,
    overwrite="true"
)
