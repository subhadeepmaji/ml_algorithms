
class DataSourceSink:
    """
    A datasource object generating data instances
    """
    def __init__(self, name):
        self.name = name

    def __init_connection__(self, host_name, host_port):
        self.host_name = host_name
        self.host_port = host_port

    def connect(self):
        """
        connect to the underlying concrete data source
        :return:
        """
        pass

    def form_cursor(self):
        """
        form a cursor object to read from
        :return:
        """
        pass

    def read_item(self):
        pass
