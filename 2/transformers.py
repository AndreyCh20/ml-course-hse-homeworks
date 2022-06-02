

import pandas as pd
from haversine import haversine
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TraficHoursTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.traffic_d_h = []
        self.no_traffic_d_h = []

    def fit(self, df):
        df_temp = df.copy()

        df_temp['av_speed_loglog'] = (
            df_temp['log_haversine'] /
            df_temp['log_trip_duration'])

        gb_with_traffic = pd.DataFrame(
            df_temp.groupby(by=['day_of_week', 'hour'])['av_speed_loglog'].median())

        gb_with_traffic['av_speed_loglog'] = (
            gb_with_traffic['av_speed_loglog'] <
            gb_with_traffic['av_speed_loglog'].quantile(0.3))

        gb_with_traffic.reset_index(level=['day_of_week', 'hour'], inplace=True)

        self.traffic_d_h = gb_with_traffic[
            gb_with_traffic['av_speed_loglog']].apply(
                lambda x: (x.day_of_week, x.hour), axis=1).values

        gb_no_traffic = pd.DataFrame(
            df_temp.groupby(by=['day_of_week', 'hour'])['av_speed_loglog'].median())

        gb_no_traffic['av_speed_loglog'] = (
            gb_no_traffic['av_speed_loglog'] >
            gb_no_traffic['av_speed_loglog'].quantile(0.7))

        gb_no_traffic.reset_index(level=['day_of_week', 'hour'], inplace=True)

        self.no_traffic_d_h = gb_no_traffic[
            gb_no_traffic['av_speed_loglog']].apply(
                lambda x: (x.day_of_week, x.hour), axis=1).values
        return self

    def transform(self, df):
        return df.assign(
            trafic=(df['day_of_week'].isin({x[0] for x in self.traffic_d_h}) &
                    df['hour'].isin({x[1] for x in self.traffic_d_h})),
            no_trafic=(df['day_of_week'].isin({x[0] for x in self.no_traffic_d_h}) &
                       df['hour'].isin({x[1] for x in self.no_traffic_d_h})))


class MapGridTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, hor_bins, ver_bins):
        self.hor_bins = hor_bins
        self.ver_bins = ver_bins
        
        self.pickup_long = []
        self.pickup_lat = []
        self.dropoff_long = []
        self.dropoff_lat = []

    def fit(self, df):
        x0 = df['pickup_longitude'].quantile(0.01)
        xn = df['pickup_longitude'].quantile(0.99)
        self.pickup_long.append(x0)
        for i in range(1, self.hor_bins):
            xi = ((xn - x0) / self.hor_bins) * i + x0
            self.pickup_long.append(xi)
        self.pickup_long.append(xn)

        x0 = df['pickup_latitude'].quantile(0.01)
        xn = df['pickup_latitude'].quantile(0.99)
        self.pickup_lat.append(x0)
        for i in range(1, self.ver_bins):
            xi = ((xn - x0) / self.ver_bins) * i + x0
            self.pickup_lat.append(xi)
        self.pickup_lat.append(xn)

        x0 = df['dropoff_longitude'].quantile(0.01)
        xn = df['dropoff_longitude'].quantile(0.99)
        self.dropoff_long.append(x0)
        for i in range(1, self.hor_bins):
            xi = ((xn - x0) / self.hor_bins) * i + x0
            self.dropoff_long.append(xi)
        self.dropoff_long.append(xn)

        x0 = df['dropoff_latitude'].quantile(0.01)
        xn = df['dropoff_latitude'].quantile(0.99)
        self.dropoff_lat.append(x0)
        for i in range(1, self.ver_bins):
            xi = ((xn - x0) / self.ver_bins) * i + x0
            self.dropoff_lat.append(xi)
        self.dropoff_lat.append(xn)

        return self

    def __subtransform(self, lng, lat, long_bounds, lat_bounds):
        if (lng < long_bounds[0] or lat < lat_bounds[0] or
           lng > long_bounds[-1] or lat > lat_bounds[-1]):
            square = -1
        else:
            square = 0
            for n, c in enumerate(long_bounds):
                if c > lng:
                    square += n - 1
                    break
            for n, c in enumerate(lat_bounds):
                if c > lat:
                    square += (n - 1) * (len(long_bounds) - 1)
                    break

        return square

    def transform(self, df):
        return df.assign(
            pickup_square=df.apply(
                lambda x: self.__subtransform(
                    x.pickup_longitude, x.pickup_latitude,
                    self.pickup_long, self.pickup_lat),
                axis=1),
            dropoff_square=df.apply(
                lambda x: self.__subtransform(
                    x.dropoff_longitude, x.dropoff_latitude,
                    self.dropoff_long, self.dropoff_lat),
                axis=1))
