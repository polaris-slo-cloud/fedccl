import os
import pickle
import threading

from src.shared.model.site import Site
from src.server.aggregator.fed_avg_dual_aggregator import FedAvgDualAggregator
from src.shared.model.model_data import ModelData
from src.shared import utils
from src.shared.model.model_meta import ModelMeta


class SiteClusterManager:
    def __init__(self, sites, data_path):
        self.data_path = data_path

        self.aggregator = FedAvgDualAggregator()

        self.sites = {}
        self.site_online_states = {}
        
        self.clusters = {}
        self.cluster_models = {}

        self.global_model = None

        # Add locks for sites, clusters and global model
        self.site_locks = {}
        self.cluster_locks = {}
        self.global_model_lock = threading.Lock()

        for site in sites:
            self.sites[site.site_id] = site

            # iterate over dict of clusters in site
            for key, value in site.clusters.items():
                if key not in self.clusters:
                    self.clusters[key] = {}
                    self.cluster_models[key] = {}
                    self.cluster_locks[key] = {}

                if value not in self.clusters[key]:
                    cluster_id = utils.get_cluster_id(key, value)
                    model_data_path = f"{self.data_path}/cluster_{cluster_id}_model_data.pkl"
                    if os.path.exists(model_data_path):
                        with open(model_data_path) as f:
                            model_data = pickle.load(f)
                    else:
                        model_data = ModelData(ModelMeta(), utils.get_model().get_weights())

                    self.clusters[key][value] = []
                    self.cluster_models[key][value] = model_data
                    self.cluster_locks[key][value] = threading.Lock()

                self.clusters[key][value].append(site.site_id)

        
        model_data_path = f"{self.data_path}/global_model_data.pkl"
        if os.path.exists(model_data_path):
            with open(model_data_path) as f:
                model_data = pickle.load(f)
        else:
            model_data = ModelData(ModelMeta(), utils.get_model().get_weights())
        
        self.global_model = model_data

    def set_site_online(self, site_id):
        self.site_online_states[site_id] = True

    def set_site_offline(self, site_id):
        self.site_online_states[site_id] = False

    def is_site_online(self, site_id):
        return self.site_online_states.get(site_id, False)
    
    def get_sites_in_cluster(self, cluster_key, cluster_value):
        cluster = self.clusters.get(cluster_key, None)
        if cluster is None:
            return []
        return cluster.get(cluster_value, None)
    
    def is_site_known(self, site_id):
        return site_id in self.sites
    
    def get_cluster_for_site(self, site_id, cluster_key):
        return self.sites[site_id].clusters[cluster_key]
    
    def update_cluster_model(self, site_id, cluster_key, model_data, model_delta_meta):
        cluster = self.get_cluster_for_site(site_id, cluster_key)
        with self.cluster_locks[cluster_key][cluster]:
            self.cluster_models[cluster_key][cluster] = self.aggregator.aggregate(self.cluster_models[cluster_key][cluster], model_data, model_delta_meta)
    
    def update_global_model(self, model_data, model_delta_meta):
        with self.global_model_lock:
            self.global_model = self.aggregator.aggregate(self.global_model, model_data, model_delta_meta)

    def get_cluster_model(self, site_id, cluster_key):
        cluster = self.get_cluster_for_site(site_id, cluster_key)
        with self.cluster_locks[cluster_key][cluster]:
            return self.cluster_models[cluster_key][cluster]
        
    def get_global_model(self):
        with self.global_model_lock:
            return self.global_model


