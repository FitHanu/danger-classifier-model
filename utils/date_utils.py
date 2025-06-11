from datetime import datetime, timedelta, timezone


def get_formated_date_as_string():
    gmt_plus_7 = timezone(timedelta(hours=7))
    folder_name = datetime.now(gmt_plus_7).strftime("%Y-%m-%d-%H-%M")
    return folder_name