import sys
import base64
from databricks_api import DatabricksAPI

prod_token = sys.argv[1]
prod_host = sys.argv[2]
notebook_name = sys.argv[3]

db = DatabricksAPI(
    host=prod_host,
    token=prod_token
)

with open("{}".format(notebook_name)) as file:
    data = file.read()

pipeline_main_file_name = "/Shared/Accelerators/{}-auto-added".format(notebook_name.replace(".py", ""))

encodedBytes = base64.b64encode(data.encode("utf-8"))
encodedStr = str(encodedBytes, "utf-8")

db.workspace.import_workspace(
    pipeline_main_file_name,
    format="SOURCE",
    language="PYTHON",
    content=encodedStr,
    overwrite="true"
)
