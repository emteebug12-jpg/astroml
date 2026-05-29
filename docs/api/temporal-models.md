# Temporal GNN Models API Documentation

## Overview

Temporal Graph Neural Networks (GNNs) extend traditional graph neural networks to handle time-evolving graph structures. These models are particularly useful for analyzing blockchain data where transactions occur over time and temporal patterns are crucial for understanding network dynamics.

## Table of Contents

1. [Temporal Encoding](#temporal-encoding)
2. [Temporal Attention](#temporal-attention)
3. [TemporalGCN](#temporalgcn)
4. [TemporalGraphSAGE](#temporalgraphsage)
5. [TemporalGAT](#temporalgat)
6. [TemporalEdgeConv](#temporaledgeconv)
7. [TemporalGraphTransformer](#temporalgraphtransformer)
8. [Model Factory](#model-factory)
9. [Temporal Data Processing](#temporal-data-processing)
10. [Training Utilities](#training-utilities)

## Temporal Encoding

### TemporalEncoding

Temporal encoding converts timestamps into meaningful features using sinusoidal functions.

```python
class TemporalEncoding(nn.Module):
    def __init__(self, temporal_dim: int, max_time: float = 1000.0)
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `temporal_dim` | `int` | Yes | - | Dimension of temporal encoding |
| `max_time` | `float` | No | `1000.0` | Maximum time value for normalization |

#### Methods

##### forward()

Encode timestamps to temporal features.

**Parameters:**
- `timestamps` (torch.Tensor): Tensor of shape [num_nodes] with timestamp values

**Returns:** `torch.Tensor` - Temporal encoding tensor of shape [num_nodes, temporal_dim]

**Example:**
```python
import torch
from astroml.models.temporal import TemporalEncoding

# Create temporal encoder
encoder = TemporalEncoding(temporal_dim=32)

# Encode timestamps
timestamps = torch.tensor([1.0, 2.0, 3.0, 4.0])
temporal_features = encoder(timestamps)

print(f"Temporal features shape: {temporal_features.shape}")  # [4, 32]
```

## Temporal Attention

### TemporalAttention

Temporal attention mechanism for time-aware node representations.

```python
class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, temporal_dim: int, heads: int = 8)
    def forward(self, x: torch.Tensor, temporal_encoding: torch.Tensor) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `temporal_dim` | `int` | Yes | - | Temporal feature dimension |
| `heads` | `int` | No | `8` | Number of attention heads |

#### Methods

##### forward()

Apply temporal attention to node features.

**Parameters:**
- `x` (torch.Tensor): Node features [num_nodes, input_dim]
- `temporal_encoding` (torch.Tensor): Temporal features [num_nodes, temporal_dim]

**Returns:** `torch.Tensor` - Attention-enhanced features [num_nodes, input_dim]

**Example:**
```python
from astroml.models.temporal import TemporalAttention

# Create attention layer
attention = TemporalAttention(input_dim=64, temporal_dim=32, heads=8)

# Apply attention
node_features = torch.randn(100, 64)
temporal_encoding = torch.randn(100, 32)

enhanced_features = attention(node_features, temporal_encoding)
print(f"Enhanced features shape: {enhanced_features.shape}")  # [100, 64]
```

## TemporalGCN

### Overview

TemporalGCN extends the standard Graph Convolutional Network with temporal encoding and attention mechanisms.

```python
class TemporalGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        time_encoding: str = "sinusoidal",
        use_attention: bool = True
    )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None,
        node_time: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `temporal_dim` | `int` | No | `32` | Temporal encoding dimension |
| `dropout` | `float` | No | `0.5` | Dropout rate |
| `time_encoding` | `str` | No | `"sinusoidal"` | Time encoding type |
| `use_attention` | `bool` | No | `True` | Use temporal attention |

#### Methods

##### forward()

Forward pass with temporal information.

**Parameters:**
- `x` (torch.Tensor): Node features [num_nodes, input_dim]
- `edge_index` (torch.Tensor): Edge indices [2, num_edges]
- `edge_time` (Optional[torch.Tensor]): Edge timestamps [num_edges]
- `node_time` (Optional[torch.Tensor]): Node timestamps [num_nodes]
- `edge_attr` (Optional[torch.Tensor]): Edge attributes [num_edges, edge_attr_dim]

**Returns:** `torch.Tensor` - Node predictions [num_nodes, output_dim]

**Example:**
```python
from astroml.models.temporal import TemporalGCN

# Create TemporalGCN model
model = TemporalGCN(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    dropout=0.5
)

# Forward pass
node_features = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
node_times = torch.rand(100)
edge_times = torch.rand(3)

predictions = model(node_features, edge_index, node_time=node_times, edge_time=edge_times)
print(f"Predictions shape: {predictions.shape}")  # [100, 2]
```

## TemporalGraphSAGE

### Overview

TemporalGraphSAGE incorporates temporal information into the GraphSAGE architecture for inductive learning on time-evolving graphs.

```python
class TemporalGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        num_layers: int = 2,
        aggregator: str = "mean"
    )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `temporal_dim` | `int` | No | `32` | Temporal encoding dimension |
| `dropout` | `float` | No | `0.5` | Dropout rate |
| `num_layers` | `int` | No | `2` | Number of SAGE layers |
| `aggregator` | `str` | No | `"mean"` | Aggregation function |

#### Example

```python
from astroml.models.temporal import TemporalGraphSAGE

# Create TemporalGraphSAGE model
model = TemporalGraphSAGE(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    aggregator="mean"
)

# Forward pass
node_features = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
node_times = torch.rand(100)

predictions = model(node_features, edge_index, node_time=node_times)
print(f"Predictions shape: {predictions.shape}")
```

## TemporalGAT

### Overview

TemporalGAT extends the Graph Attention Network with temporal encoding and time-aware attention mechanisms.

```python
class TemporalGAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        heads: int = 8,
        concat: bool = True
    )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `temporal_dim` | `int` | No | `32` | Temporal encoding dimension |
| `dropout` | `float` | No | `0.5` | Dropout rate |
| `heads` | `int` | No | `8` | Number of attention heads |
| `concat` | `bool` | No | `True` | Concatenate multi-head outputs |

#### Example

```python
from astroml.models.temporal import TemporalGAT

# Create TemporalGAT model
model = TemporalGAT(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    heads=8
)

# Forward pass
node_features = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
node_times = torch.rand(100)

predictions = model(node_features, edge_index, node_time=node_times)
print(f"Predictions shape: {predictions.shape}")
```

## TemporalEdgeConv

### Overview

TemporalEdgeConv implements edge convolution with temporal information for dynamic edge features.

```python
class TemporalEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_dim: int = 32,
        aggr: str = "max"
    )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `in_channels` | `int` | Yes | - | Input feature channels |
| `out_channels` | `int` | Yes | - | Output feature channels |
| `temporal_dim` | `int` | No | `32` | Temporal encoding dimension |
| `aggr` | `str` | No | `"max"` | Aggregation method |

#### Example

```python
from astroml.models.temporal import TemporalEdgeConv

# Create TemporalEdgeConv layer
edge_conv = TemporalEdgeConv(
    in_channels=64,
    out_channels=32,
    temporal_dim=32
)

# Forward pass
node_features = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
edge_times = torch.rand(3)

updated_features = edge_conv(node_features, edge_index, edge_times)
print(f"Updated features shape: {updated_features.shape}")  # [100, 32]
```

## TemporalGraphTransformer

### Overview

TemporalGraphTransformer combines transformer architecture with graph structure for temporal graph analysis.

```python
class TemporalGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        temporal_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dim` | `int` | Yes | - | Hidden layer dimension |
| `output_dim` | `int` | Yes | - | Output dimension |
| `temporal_dim` | `int` | No | `32` | Temporal encoding dimension |
| `num_heads` | `int` | No | `8` | Number of attention heads |
| `num_layers` | `int` | No | `3` | Number of transformer layers |
| `dropout` | `float` | No | `0.1` | Dropout rate |

#### Example

```python
from astroml.models.temporal import TemporalGraphTransformer

# Create TemporalGraphTransformer model
model = TemporalGraphTransformer(
    input_dim=64,
    hidden_dim=128,
    output_dim=2,
    temporal_dim=32,
    num_heads=8,
    num_layers=3
)

# Forward pass
node_features = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
node_times = torch.rand(100)

predictions = model(node_features, edge_index, node_time=node_times)
print(f"Predictions shape: {predictions.shape}")
```

## Model Factory

### TemporalModelFactory

Factory class for creating temporal GNN models with consistent configuration.

```python
class TemporalModelFactory:
    @staticmethod
    def create_temporal_gcn(config) -> TemporalGCN
    
    @staticmethod
    def create_temporal_sage(config) -> TemporalGraphSAGE
    
    @staticmethod
    def create_temporal_gat(config) -> TemporalGAT
    
    @staticmethod
    def create_temporal_transformer(config) -> TemporalGraphTransformer
```

#### Methods

##### create_temporal_gcn()

Create TemporalGCN model from configuration.

**Parameters:**
- `config`: Configuration object with model parameters

**Returns:** `TemporalGCN` - Configured model

##### create_temporal_sage()

Create TemporalGraphSAGE model from configuration.

**Parameters:**
- `config`: Configuration object with model parameters

**Returns:** `TemporalGraphSAGE` - Configured model

##### create_temporal_gat()

Create TemporalGAT model from configuration.

**Parameters:**
- `config`: Configuration object with model parameters

**Returns:** `TemporalGAT` - Configured model

##### create_temporal_transformer()

Create TemporalGraphTransformer model from configuration.

**Parameters:**
- `config`: Configuration object with model parameters

**Returns:** `TemporalGraphTransformer` - Configured model

**Example:**
```python
from astroml.models.temporal import TemporalModelFactory

# Configuration
config = {
    'input_dim': 64,
    'hidden_dims': [128, 64],
    'output_dim': 2,
    'temporal_dim': 32,
    'dropout': 0.5
}

# Create models
gcn_model = TemporalModelFactory.create_temporal_gcn(config)
sage_model = TemporalModelFactory.create_temporal_sage(config)
gat_model = TemporalModelFactory.create_temporal_gat(config)
transformer_model = TemporalModelFactory.create_temporal_transformer(config)
```

## Temporal Data Processing

### TemporalDataProcessor

Utilities for processing temporal graph data.

```python
class TemporalDataProcessor:
    def __init__(self, time_window: Optional[int] = None)
    
    def normalize_timestamps(
        self,
        timestamps: Union[List[datetime], np.ndarray, torch.Tensor],
        reference_time: Optional[datetime] = None
    ) -> torch.Tensor
    
    def create_time_windows(
        self,
        timestamps: torch.Tensor,
        window_size: int,
        overlap: float = 0.5
    ) -> List[Tuple[float, float]]
    
    def filter_edges_by_time(
        self,
        edge_index: torch.Tensor,
        edge_times: torch.Tensor,
        time_window: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]
    
    def compute_temporal_features(
        self,
        timestamps: torch.Tensor,
        features: torch.Tensor,
        window_size: int = 10
    ) -> torch.Tensor
```

#### Methods

##### normalize_timestamps()

Normalize timestamps to relative values.

**Parameters:**
- `timestamps`: List or array of timestamps
- `reference_time`: Reference time for normalization

**Returns:** `torch.Tensor` - Normalized timestamps

##### create_time_windows()

Create overlapping time windows.

**Parameters:**
- `timestamps`: Normalized timestamps
- `window_size`: Size of each window
- `overlap`: Overlap between windows (0 to 1)

**Returns:** `List[Tuple[float, float]]` - List of (start_time, end_time) tuples

##### filter_edges_by_time()

Filter edges by time window.

**Parameters:**
- `edge_index`: Edge indices [2, num_edges]
- `edge_times`: Edge timestamps [num_edges]
- `time_window`: (start_time, end_time) tuple

**Returns:** `Tuple[torch.Tensor, torch.Tensor]` - Filtered edge_index and edge_times

##### compute_temporal_features()

Compute temporal features for nodes.

**Parameters:**
- `timestamps`: Node timestamps [num_nodes]
- `features`: Node features [num_nodes, feature_dim]
- `window_size`: Window size for temporal aggregation

**Returns:** `torch.Tensor` - Temporal features [num_nodes, temporal_feature_dim]

**Example:**
```python
from astroml.utils.temporal import TemporalDataProcessor

# Create processor
processor = TemporalDataProcessor()

# Normalize timestamps
timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
normalized_times = processor.normalize_timestamps(timestamps)

# Create time windows
windows = processor.create_time_windows(normalized_times, window_size=0.2, overlap=0.1)
print(f"Time windows: {windows}")
```

### TemporalGraphBuilder

Build temporal graphs from transaction data.

```python
class TemporalGraphBuilder:
    def __init__(self, time_window_days: int = 30)
    
    def build_temporal_graph(
        self,
        transactions: List[Dict],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]
```

#### Methods

##### build_temporal_graph()

Build temporal graph from transaction data.

**Parameters:**
- `transactions`: List of transaction dictionaries
- `node_features`: Optional pre-computed node features

**Returns:** `Dict[str, torch.Tensor]` - Graph data dictionary

**Example:**
```python
from astroml.utils.temporal import TemporalGraphBuilder

# Create builder
builder = TemporalGraphBuilder(time_window_days=30)

# Sample transactions
transactions = [
    {
        'source_account': 'GABC...',
        'target_account': 'GDEF...',
        'amount': 100.0,
        'timestamp': 1640995200,  # Unix timestamp
        'operation_type': 'payment'
    },
    # ... more transactions
]

# Build temporal graph
graph_data = builder.build_temporal_graph(transactions)

print(f"Graph keys: {graph_data.keys()}")
print(f"Number of nodes: {graph_data['num_nodes']}")
print(f"Edge index shape: {graph_data['edge_index'].shape}")
```

## Training Utilities

### TemporalTrainer

Trainer for temporal GNN models with temporal-specific features.

```python
class TemporalTrainer:
    def __init__(self, config: TemporalTrainingConfig)
    
    def train(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: Optional[List[Dict[str, torch.Tensor]]] = None,
        val_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]
    
    def evaluate(
        self,
        test_graphs: List[Dict[str, torch.Tensor]],
        test_labels: torch.Tensor
    ) -> Dict[str, float]
```

#### Configuration

```python
@dataclass
class TemporalTrainingConfig:
    model_type: str = "temporal_gcn"
    input_dim: int = 64
    hidden_dims: List[int] = None
    output_dim: int = 2
    temporal_dim: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    dropout: float = 0.5
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    temporal_consistency_weight: float = 0.1
    augmentation_enabled: bool = True
    augmentation_prob: float = 0.3
```

#### Example

```python
from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

# Create configuration
config = TemporalTrainingConfig(
    model_type="temporal_gcn",
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    epochs=50,
    temporal_consistency_weight=0.1
)

# Create trainer
trainer = TemporalTrainer(config)

# Train model
history = trainer.train(train_graphs, train_labels, val_graphs, val_labels)

# Evaluate
test_results = trainer.evaluate(test_graphs, test_labels)

print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
print(f"Temporal AUC: {test_results['temporal_auc']:.4f}")
```

### TemporalHyperparameterSearch

Hyperparameter optimization for temporal models.

```python
class TemporalHyperparameterSearch:
    def __init__(self, search_space: Dict[str, List[Any]])
    
    def random_search(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: List[Dict[str, torch.Tensor]],
        val_labels: torch.Tensor,
        n_trials: int = 50
    ) -> Dict[str, Any]
```

#### Example

```python
from astroml.training.temporal import TemporalHyperparameterSearch, DEFAULT_SEARCH_SPACE

# Create search
search = TemporalHyperparameterSearch(DEFAULT_SEARCH_SPACE)

# Perform random search
results = search.random_search(
    train_graphs, train_labels,
    val_graphs, val_labels,
    n_trials=20
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

### TemporalExperiment

Experiment management for temporal models.

```python
class TemporalExperiment:
    def __init__(self, name: str, config: TemporalTrainingConfig)
    
    def run_experiment(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: List[Dict[str, torch.Tensor]],
        val_labels: torch.Tensor,
        test_graphs: List[Dict[str, torch.Tensor]],
        test_labels: torch.Tensor
    ) -> Dict[str, Any]
    
    def save_results(self, filepath: str)
```

#### Example

```python
from astroml.training.temporal import TemporalExperiment, TemporalTrainingConfig

# Create experiment
config = TemporalTrainingConfig(
    model_type="temporal_gcn",
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2
)

experiment = TemporalExperiment("fraud_detection_temporal_gcn", config)

# Run experiment
results = experiment.run_experiment(
    train_graphs, train_labels,
    val_graphs, val_labels,
    test_graphs, test_labels
)

# Save results
experiment.save_results("temporal_gcn_experiment.json")
```

## Usage Examples

### Complete Temporal GNN Pipeline

```python
from astroml.models.temporal import TemporalGCN
from astroml.utils.temporal import TemporalGraphBuilder, TemporalDataProcessor
from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

# 1. Build temporal graph from transactions
builder = TemporalGraphBuilder(time_window_days=30)
graph_data = builder.build_temporal_graph(transactions)

# 2. Create model
model = TemporalGCN(
    input_dim=graph_data['node_features'].shape[1],
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32
)

# 3. Train model
config = TemporalTrainingConfig(
    model_type="temporal_gcn",
    input_dim=graph_data['node_features'].shape[1],
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    epochs=100
)

trainer = TemporalTrainer(config)
history = trainer.train([graph_data], labels)

# 4. Evaluate
test_results = trainer.evaluate([test_graph_data], test_labels)

print(f"Final Results:")
print(f"  Test Accuracy: {test_results['test_accuracy']:.4f}")
print(f"  Temporal AUC: {test_results['temporal_auc']:.4f}")
print(f"  Temporal Accuracy: {test_results['temporal_accuracy']:.4f}")
```

### Multi-Model Comparison

```python
from astroml.models.temporal import TemporalModelFactory
from astroml.training.temporal import create_temporal_experiment_suite

# Compare different temporal models
results = create_temporal_experiment_suite(
    train_graphs, train_labels,
    val_graphs, val_labels,
    test_graphs, test_labels
)

# Print comparison
for i, result in enumerate(results):
    config = result['config']
    test_results = result['test_results']
    
    print(f"Model {i+1} ({config.model_type}):")
    print(f"  Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  Temporal AUC: {test_results['temporal_auc']:.4f}")
```

## Performance Considerations

### Temporal Encoding Optimization

```python
# Use learnable temporal encoding for better performance
model = TemporalGCN(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32,
    time_encoding="learnable"  # Better for specific domains
)
```

### Batch Processing

```python
# Process multiple time windows in batches
processor = TemporalDataProcessor()
windows = processor.create_time_windows(timestamps, window_size=0.1, overlap=0.05)

for window_start, window_end in windows:
    # Filter data for this window
    window_data = processor.filter_edges_by_time(edge_index, edge_times, (window_start, window_end))
    
    # Process window
    predictions = model(node_features, window_data['edge_index'], node_time, window_data['edge_times'])
```

### Memory Management

```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x, edge_index, node_time, edge_time):
    return checkpoint(model.forward, x, edge_index, node_time, edge_time)
```

## Error Handling

### Common Issues and Solutions

#### 1. Temporal Dimension Mismatch
```python
# Ensure temporal dimensions match
if temporal_encoding.size(1) != model.temporal_dim:
    temporal_encoding = F.adaptive_avg_pool1d(
        temporal_encoding.t().unsqueeze(0), model.temporal_dim
    ).t().squeeze(0)
```

#### 2. Time Range Issues
```python
# Normalize timestamps to [0, 1] range
timestamps = processor.normalize_timestamps(timestamps)
timestamps = torch.clamp(timestamps, 0.0, 1.0)
```

#### 3. Memory Issues
```python
# Use smaller batch sizes for large graphs
config.batch_size = min(config.batch_size, graph_data['num_nodes'] // 10)
```

---

This comprehensive documentation covers all temporal GNN models in AstroML, from basic temporal encoding to advanced transformer architectures, with complete usage examples and best practices for blockchain data analysis.
