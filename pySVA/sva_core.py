__all__ = ['constructorSVA']

# Do imports
import xarray as xr
import xugrid as xu

class constructorSVA:
    def __init__(self, input_file, data_description, **kwargs):

        if isinstance(input_file, str):
            try:
                self._read(filename=input_file)
            except (OSError, IOError, RuntimeError):
                raise IOError("Unable to read file.")
        elif isinstance(input_file, xr.Dataset):
            self.ds = input_file
        elif isinstance(input_file, xu.core.wrap.UgridDataset):
            self.ds = input_file
        elif isinstance(input_file, xr.DataArray):
            self.ds = input_file
        else:
            raise IOError("Please provide xr.Dataset, xr.DataArray, or a file path to a netCDF.")

        self._setup(data=data_description)
        self.transport = None
        self.tracer = None
        self.tracer_variance = None

    def _read(self, filename, **kwargs):
        self.ds = xr.open_dataset(filename, use_cftime=True, **kwargs)

    def _setup(self, data):
        """Iterates over keys in dictionary. Handles 4d-data, if one argument is left empty, dummy dimension will be created.
        Args:
            data (dict): Dictionary that describes geospatial dimensions of the dataset.
        """
        for dimension in data.keys():
            if data[dimension] is None:
                self.ds = self.ds.expand_dims(dimension)
            else:
                self.ds = self.ds.rename({data[dimension]: dimension})

        self.ds = self.ds.transpose("time",
                                    "depth",
                                    "lat",
                                    "lon")
