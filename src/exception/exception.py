import sys

def error_message_detail(error, error_detail:sys):

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    return f"""
    Error occurred in script:
    [{file_name}]
    line number [{exc_tb.tb_lineno}]
    error message [{str(error)}]
    """