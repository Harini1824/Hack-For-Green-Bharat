import pathway as pw

class WasteSchema(pw.Schema):
    waste_type: str

input_table = pw.io.csv.read(
    "waste_log.csv",
    schema=WasteSchema
)

result = input_table.groupby(input_table.waste_type).reduce(
    count=pw.reducers.count()
)

pw.io.csv.write(result, "dashboard_output.csv")

pw.run()