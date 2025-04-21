from flask import Flask, render_template_string, request, send_file, jsonify
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend
import matplotlib.pyplot as plt
import io, base64, random
import math
import json
import os
from datetime import datetime
import uuid
import re

# reuse your generator
from gaucher_graph_experiments import gen_example, gen_example_er, gen_example_ws, gen_example_ba

app = Flask(__name__)

n=40
n_byz=5
mu_min = 0 # min mu_2 of honest nodes
f = 4 # maximum number of byzantine nodes in the neighborhood of any honest node
default_num_graphs = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Store generated graphs for the current session
SESSION_GRAPHS = {}

# list of (generator, params) for various graph types
graph_builders = [
    (gen_example, dict(name="Expander-k6", n=n, n_byz=n_byz, mu_min=mu_min, f=f, k=6, seed=None)),
    (gen_example, dict(name="Denser-k8",   n=n, n_byz=n_byz, mu_min=mu_min, f=f, k=8, seed=None)),
    (gen_example, dict(name="Sparse-k4",   n=n, n_byz=n_byz, mu_min=mu_min, f=f, k=4, seed=None)),
    (gen_example_er, dict(name="ER-p0.2", n=n, n_byz=n_byz, p=0.2, mu_min=mu_min, f=f, seed=None)),
    (gen_example_er, dict(name="ER-p0.5", n=n, n_byz=n_byz, p=0.5, mu_min=mu_min, f=f, seed=None)),
    (gen_example_ws, dict(name="WS-k4-p0.1", n=n, n_byz=n_byz, k=4, p=0.1, mu_min=mu_min, f=f, seed=None)),
    (gen_example_ws, dict(name="WS-k6-p0.3", n=n, n_byz=n_byz, k=6, p=0.3, mu_min=mu_min, f=f, seed=None)),
    (gen_example_ba, dict(name="BA-m2", n=n, n_byz=n_byz, m=2, mu_min=mu_min, f=f, seed=None)),
    (gen_example_ba, dict(name="BA-m3", n=n, n_byz=n_byz, m=3, mu_min=mu_min, f=f, seed=None)),
]

TEMPLATE = """
<html>
  <head>
    <title>Random Byzantine Graphs</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
      body {
        font-family: Arial, sans-serif;
      }
      .form-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
      }
      .form-group {
        display: inline-block;
        margin-right: 15px;
        margin-bottom: 10px;
      }
      .form-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
      }
      .form-group input {
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        width: 80px;
      }
      .submit-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
      }
      .submit-btn:hover {
        background-color: #45a049;
      }
      .graph-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 20px;
        padding: 20px;
      }
      .graph-card {
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 8px;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      .export-btn {
        background-color: #4CAF50;
        color: white;
        padding: 8px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
      }
      .export-btn:hover {
        background-color: #45a049;
      }
      .status {
        margin-top: 10px;
        padding: 5px;
        border-radius: 5px;
        display: none;
      }
      .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
      }
      .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
      }
      .info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
      }
    </style>
  </head>
  <body>
    <h1 style="text-align: center; margin: 20px 0;">Byzantine Graph Generator</h1>
    <div class="form-container">
      <form action="/" method="get">
        <div class="form-group">
          <label for="num_graphs">Number of graphs:</label>
          <input type="number" id="num_graphs" name="num_graphs" value="{{ num_graphs }}" min="1" max="100">
        </div>
        <div class="form-group">
          <label for="n">Total nodes (n):</label>
          <input type="number" id="n" name="n" value="{{ n }}" min="1">
        </div>
        <div class="form-group">
          <label for="n_byz">Byzantine nodes (n_byz):</label>
          <input type="number" id="n_byz" name="n_byz" value="{{ n_byz }}" min="0" max="{{ n }}">
        </div>
        <div class="form-group">
          <label for="f">Byzantine budget (f):</label>
          <input type="number" id="f" name="f" value="{{ f }}" min="0" max="{{ n_byz }}">
        </div>
        <button type="submit" class="submit-btn">Generate Graphs</button>
      </form>
    </div>
    <div class="graph-grid">
    {% for graph in graphs %}
      <div class="graph-card">
        <h2>{{ graph.stats['example'] }}</h2>
        <img src="data:image/png;base64,{{ graph.img }}" style="width: 100%; max-width: 400px;"/>
        <h3>Graph statistics</h3>
        <ul>
          <li>n_hon: {{ graph.stats['n_hon'] }}</li>
          <li>n_byz: {{ graph.stats['n_byz'] }}</li>
          <li>f (byzantine budget): {{ graph.stats['f'] }}</li>
          <li>μ₂(G_H): {{ graph.stats['mu2'] }}</li>
          <li>max b(i): {{ graph.stats['max_b'] }}</li>
          <li>density (honest): {{ graph.stats['density_honest'] }}</li>
          <li>density (full): {{ graph.stats['density_full'] }}</li>
          <li>max degree: {{ graph.stats['max_degree'] }}</li>
          <li>Is 2b < μ₂:
            <span style="color:{% if 2*graph.stats['max_b'] < graph.stats['mu2'] %}blue{% else %}red{% endif %};">
              {{ 2*graph.stats['max_b'] < graph.stats['mu2'] }}
            </span>
          </li>
          <li>Is 4b < μ₂:
            <span style="color:{% if 4*graph.stats['max_b'] < graph.stats['mu2'] %}blue{% else %}red{% endif %};">
              {{ 4*graph.stats['max_b'] < graph.stats['mu2'] }}
            </span>
          </li>
        </ul>
        <form action="/export/{{ graph.id }}" method="get">
          <button type="submit" class="export-btn">Export Graph</button>
        </form>
        <div class="status" id="status-{{ graph.id }}"></div>
      </div>
    {% endfor %}
    </div>
    
    <script>
      $(document).ready(function() {
        // Form validation for export forms
        $('form').submit(function(e) {
          if ($(this).attr('action').startsWith('/export')) {
            e.preventDefault();
            
            var formAction = $(this).attr('action');
            var graphId = formAction.split('/').pop();
            var statusDiv = $('#status-' + graphId);
            
            statusDiv.removeClass('success error').addClass('info').text('Exporting...').show();
            
            $.ajax({
              url: formAction,
              type: 'get',
              success: function(response) {
                if (response.success) {
                  statusDiv.removeClass('info').addClass('success').text('Exported successfully to: ' + response.path);
                } else {
                  statusDiv.removeClass('info').addClass('error').text('Export failed: ' + response.error);
                }
              },
              error: function(xhr, status, error) {
                statusDiv.removeClass('info').addClass('error').text('Export failed: ' + error);
              }
            });
          }
        });
        
        // Parameter validation
        $('#n, #n_byz, #f').on('input', function() {
          var n = parseInt($('#n').val()) || 1;
          var n_byz = parseInt($('#n_byz').val()) || 0;
          var f = parseInt($('#f').val()) || 0;
          
          // Update max attributes
          $('#n_byz').attr('max', n);
          $('#f').attr('max', n_byz);
          
          // Adjust values if needed
          if (n_byz > n) {
            $('#n_byz').val(n);
          }
          
          if (f > n_byz) {
            $('#f').val(n_byz);
          }
        });
      });
    </script>
  </body>
</html>
"""

@app.route('/')
def index():
    # Get all parameters from the request
    num_graphs = int(request.args.get('num_graphs', default_num_graphs))
    
    # Get graph parameters with defaults
    current_n = int(request.args.get('n', n))
    current_n_byz = int(request.args.get('n_byz', n_byz))
    current_f = int(request.args.get('f', f))
    
    # Validate parameters
    if current_n < current_n_byz:
        current_n = current_n_byz
    if current_f > current_n_byz:
        current_f = current_n_byz
    
    graphs = []
    
    # Clear previous session graphs
    SESSION_GRAPHS.clear()
    
    # Update the graph builders with current parameters
    current_graph_builders = []
    for gen_fn, params in graph_builders:
        # Create a copy of params and update with current values
        current_params = params.copy()
        current_params['n'] = current_n
        current_params['n_byz'] = current_n_byz
        current_params['f'] = current_f
        current_graph_builders.append((gen_fn, current_params))
    
    for _ in range(num_graphs):
        # choose random example
        gen_fn, p = random.choice(current_graph_builders)
        G, stats = gen_fn(**p)
        # unify μ₂ and max_b keys
        stats['mu2'] = stats.get('mu2', stats.get('μ₂(G_H)'))
        stats['max_b'] = stats.get('max_b', stats.get('max_b(i)'))
        
        # compute graph densities
        G_hon = G.subgraph([v for v in G.nodes() if not (isinstance(v, str) and v.startswith('B'))])
        stats['density_honest'] = round(nx.density(G_hon), 3)
        stats['density_full'] = round(nx.density(G), 3)
        
        # compute maximum degree
        stats['max_degree'] = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0

        # draw to PNG in memory
        fig = plt.figure(figsize=(5,5))
        node_colors = ['tab:red' if isinstance(v, str) and v.startswith('B') else 'tab:blue' for v in G.nodes()]
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, node_color=node_colors, with_labels=False, edge_color='gray')
        plt.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_data = base64.b64encode(buf.getvalue()).decode('ascii')
        
        # Generate unique ID for this graph
        graph_id = str(uuid.uuid4())
        graph_data = {'img': img_data, 'stats': stats, 'G': G, 'id': graph_id}
        SESSION_GRAPHS[graph_id] = graph_data
        graphs.append({'img': img_data, 'stats': stats, 'id': graph_id})

    return render_template_string(TEMPLATE, graphs=graphs, num_graphs=num_graphs, n=current_n, n_byz=current_n_byz, f=current_f)

@app.route('/export/<graph_id>')
def export_graph(graph_id):
    try:
        if graph_id not in SESSION_GRAPHS:
            return jsonify({'success': False, 'error': 'Graph not found. The page may have been refreshed.'})
        
        graph_data = SESSION_GRAPHS[graph_id]
        stats = graph_data['stats']
        img_data = graph_data['img']
        G = graph_data['G']
        
        # Clean the name for use in filename
        name = stats['example']
        clean_name = re.sub(r'[^a-z0-9]', '_', name.lower())
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{clean_name}_{timestamp}"
        
        # Save the image
        img_bytes = base64.b64decode(img_data)
        img_path = os.path.join(OUTPUT_DIR, f'{filename}.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        
        # Convert graph structure to JSON-serializable format
        graph_dict = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes with attributes
        for node in G.nodes():
            node_data = {'id': str(node)}
            # Add node attributes
            node_data['is_byzantine'] = isinstance(node, str) and node.startswith('B')
            # Add any other node attributes from the graph
            for attr, value in G.nodes[node].items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    node_data[attr] = value
            graph_dict['nodes'].append(node_data)
            
        # Add edges with attributes
        for u, v in G.edges():
            edge_data = {'source': str(u), 'target': str(v)}
            # Add any edge attributes
            for attr, value in G.edges[u, v].items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    edge_data[attr] = value
            graph_dict['edges'].append(edge_data)
        
        # Combine stats and graph structure
        export_data = {
            'stats': stats,
            'graph': graph_dict
        }
        
        # Save the stats and graph structure
        stats_path = os.path.join(OUTPUT_DIR, f'{filename}.json')
        with open(stats_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return jsonify({'success': True, 'path': f'output/{filename}.png and output/{filename}.json'})
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)