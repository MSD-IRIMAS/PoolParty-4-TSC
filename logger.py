import csv
import os

class CSV:
    def __init__(self, filename, columns):
        self.columns = columns
        
        self.file = open(filename, 'w')
        if not self.file:
            raise PermissionError(f'error: open() {filename}')
        
        self.writer = csv.DictWriter(self.file, fieldnames=columns.keys())
        if not self.writer:
            raise BlockingIOError(f'error: csv.DictWriter() {columns}')
        
        self.writer.writeheader()

    def add(self, data):
        if not self.file or not self.writer:
            raise BlockingIOError(f'error: self.add() {self.file} {self.writer}')

        # Validate the data types
        for column, col_type in self.columns.items():
            if column in data and not isinstance(data[column], col_type):
                raise ValueError(f'error: self.add() : Data for column "{column}" must be of type {col_type.__name__}.')

        self.writer.writerow(data)

    def close(self):
        self.file.close()
