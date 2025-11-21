cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)
print(f"Data cleansed and combined.\nRows: {len(data)}")

#THIS LINE COMES FROM THE CELL "DATA DRIFT ARTIFACT" IN THE OG NOTEBOOK
data.to_csv('./artifacts/training_data.csv', index=False)