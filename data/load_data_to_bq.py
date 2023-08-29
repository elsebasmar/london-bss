def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    columns_new=[]
    for column in data.columns:
        columns_new.append("_"+str(column))

    data.columns=columns_new

    PROJECT = gcp_project
    DATASET = bq_dataset
    TABLE = table

    full_table_name = f"{PROJECT}.{DATASET}.{TABLE}"

    if truncate==True:
        client = bigquery.Client()
        write_mode = "WRITE_TRUNCATE"
    else:
        client = bigquery.Client()
        write_mode = "WRITE_APPEND"

    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)

    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

    return result


####### Actually loading to bq
