from flask import Flask, request, jsonify
import os

from src.server.site_cluster_manager import SiteClusterManager
from src.shared.model.model_data import ModelData
from src.shared.model.model_meta import ModelMeta
from src.shared.model.site import Site
from src.shared.aggregation_level import AggregationLevel

class FederatedServer:
    def __init__(self, sites, port, data_path):
        self.app = Flask(__name__)

        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 
        self.app.config['DATA_UPLOAD_MAX_MEMORY_SIZE'] = 100 * 1024 * 1024

        self.scm = SiteClusterManager(sites, data_path)
        self.configure_routes()

        self.port = port

    def configure_routes(self):
        self.app.add_url_rule('/site/<site_id>/register', 'register', self.register, methods=['POST'])
        self.app.add_url_rule(f'/site/<site_id>/model/<aggregation_level>', 'upload_model', self.upload_model, methods=['POST'])
        self.app.add_url_rule('/site/<site_id>/model/<aggregation_level>', 'get_model', self.get_model, methods=['GET'])


    def register(self, site_id: str):
        if not self.scm.is_site_known(site_id):
            return jsonify({'error': 'Not allowed'}), 400
        
        self.scm.set_site_online(site_id)
        return jsonify({'message': 'Site registered successfully'}), 200
    
    def info(self):
        return jsonify(
            {
                'message': 'Server running',
                'sites_registred': len(self.scm.sites)
            }), 200

    def upload_model(self, aggregation_level: AggregationLevel, site_id: str):
        data = request.get_json()

        if not data:
            print("No data provided")
            return jsonify({'error': 'No data provided'}), 400
        elif 'model_data' not in data:
            return jsonify({'error': 'No model data provided'}), 400
        elif 'model_delta_meta' not in data:
            return jsonify({'error': 'No model delta meta provided'}), 400
        else:
            model_data = data['model_data']
            model_delta_meta = data['model_delta_meta']
            try:
                model_data = ModelData.from_json(model_data)
                model_delta_meta = ModelMeta.from_json(model_delta_meta)
            except:
                print("Model data not in correct format")
                return jsonify({'error': 'Model data not in correct format'}), 400
            
        if not self.scm.is_site_online(site_id):
            return jsonify({'error': 'Site not registered'}), 404
        

        if aggregation_level == AggregationLevel.cluster:
            cluster_key = request.args.get('cluster_key')

            if not cluster_key or self.scm.get_cluster_for_site(site_id, cluster_key) is None:
                return jsonify({'error': 'Cluster key not provided or cluster with key not found'}), 400
            
            self.scm.update_cluster_model(site_id, cluster_key, model_data, model_delta_meta)
        elif aggregation_level == AggregationLevel.global_:
            self.scm.update_global_model(model_data, model_delta_meta)
        else:
            return jsonify({'error': 'Aggregation level not supported'}), 400


        return jsonify({'message': 'Model updated successfully'}), 200


    def get_model(self, aggregation_level: AggregationLevel, site_id: str):
        if aggregation_level == AggregationLevel.cluster:
            cluster_key = request.args.get('cluster_key')

            if not cluster_key or self.scm.get_cluster_for_site(site_id, cluster_key) is None:
                return jsonify({'error': 'Cluster key not provided or cluster with key not found'}), 400

            model_data = self.scm.get_cluster_model(site_id, cluster_key)
        elif aggregation_level == AggregationLevel.global_:
            model_data = self.scm.get_global_model()
        else:
            return jsonify({'error': 'Aggregation level not supported'}), 400

        return jsonify({'model_data': model_data.to_json()}), 200

    def run(self, debug=True):
        self.app.run(host='0.0.0.0', debug=debug, port=self.port)

