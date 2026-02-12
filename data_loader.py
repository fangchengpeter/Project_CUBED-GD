import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
import os

NUM_WORKERS = min(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)) // 2, 8)
PIN_MEMORY = True

def load_data_with_validation_split(config):
    """
    Load and distribute data with proper train/validation/test splits.
    
    This implementation creates a clean separation between:
    - Training data (80% of original training set): Distributed among nodes for training
    - Validation data (20% of original training set): Shared across all nodes for monitoring
    - Test data (original test set): Reserved for final evaluation only
    
    This approach prevents data leakage and provides honest performance assessment.
    """
    try:
        # Load the base datasets with appropriate transforms
        if config.dataset == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # Load the full training set and the test set
            full_trainset = torchvision.datasets.MNIST(root=config.data_dir, train=True,
                                                      download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root=config.data_dir, train=False,
                                               download=True, transform=transform)
                                               
        elif config.dataset == "cifar10":
            # Use data augmentation only for training, not for validation
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # Validation and test use the same transform (no augmentation)
            transform_eval = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            # Load full training set with augmentation for training portion
            full_trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                                        download=True, transform=transform_train)
            # Load test set
            testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                                 download=True, transform=transform_eval)
            
            # We'll need to create a separate validation set with eval transforms
            # This is handled below in the splitting logic
        
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")

        print(f"Loaded {len(full_trainset)} training samples and {len(testset)} test samples for {config.dataset}.")
        
        # Split the training set into actual training (80%) and validation (20%)
        # We use stratified splitting to maintain class balance in both sets
        train_indices, val_indices = train_test_split(
            range(len(full_trainset)),
            test_size=0.2,  # 20% for validation
            random_state=config.seed,  # Ensure reproducible splits
            stratify=[full_trainset[i][1] for i in range(len(full_trainset))]  # Stratify by class labels
        )
        
        print(f"Split training data: {len(train_indices)} for training, {len(val_indices)} for validation")
        
        # Create training subset (this will be distributed among nodes)
        train_subset = torch.utils.data.Subset(full_trainset, train_indices)
        
        # Create validation subset
        # For CIFAR-10, we need to apply eval transforms to validation data
        if config.dataset == "cifar10":
            # Create a separate dataset for validation with eval transforms
            val_dataset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                                     download=False, transform=transform_eval)
            val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        else:
            # For MNIST, we can use the same transforms
            val_subset = torch.utils.data.Subset(full_trainset, val_indices)
        
        # Now distribute the training data among nodes
        trainloaders = []
        all_train_indices = list(range(len(train_subset)))
        np.random.shuffle(all_train_indices)  # Shuffle for random distribution
        
        loader_batch_size = getattr(config, 'loader_batch_size', 64)
        print(f"Using DataLoader batch size: {loader_batch_size}")
        
        # Calculate how to distribute training data among nodes
        samples_per_node = len(train_subset) // config.num_nodes
        remaining_samples = len(train_subset) % config.num_nodes
        current_pos = 0
        
        print(f"Distributing {len(train_subset)} training samples among {config.num_nodes} nodes")
        
        for i in range(config.num_nodes):
            # Calculate subset size for this node (distribute remainder evenly)
            node_subset_size = samples_per_node + (1 if i < remaining_samples else 0)
            node_indices = all_train_indices[current_pos: current_pos + node_subset_size]
            current_pos += node_subset_size
            
            print(f"Node {i}: {len(node_indices)} training samples")
            
            if not node_indices:
                print(f"Warning: Node {i} received no training samples.")
                empty_dataset = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
                trainloaders.append(torch.utils.data.DataLoader(
                    empty_dataset, 
                    batch_size=loader_batch_size, 
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY
                ))
                continue
            
            # Create subset of the training subset for this node
            node_subset = torch.utils.data.Subset(train_subset, node_indices)
            
            # Create DataLoader for this node with our stability fixes
            trainloaders.append(torch.utils.data.DataLoader(
                node_subset,
                batch_size=loader_batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=(NUM_WORKERS > 0),
                drop_last=False
            ))

        
        # Create validation loader (shared across all nodes for monitoring)
        validationloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=loader_batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=NUM_WORKERS,  # Same stability fixes
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
            drop_last=False
        )
        
        # Create test loader (for final evaluation only)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=loader_batch_size,
            shuffle=False,  # Never shuffle test data
            num_workers=NUM_WORKERS,  # Same stability fixes
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
            drop_last=False
        )
        
        print(f"Data distribution complete:")
        print(f"  - Training: {len(train_subset)} samples distributed among {config.num_nodes} nodes")
        print(f"  - Validation: {len(val_subset)} samples (shared)")
        print(f"  - Test: {len(testset)} samples (reserved for final evaluation)")
        print(f"DataLoader configuration: num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY} (optimized for stability)")
        
        return trainloaders, validationloader, testloader
        
    except Exception as e:
        print(f"Error loading data for {config.dataset}: {str(e)}")
        raise

def load_data_without_validation(config):
    """
    Load and distribute data into only training and test sets.
    No validation set is used.
    """
    if config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root=config.data_dir, train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=config.data_dir, train=False,
                                             download=True, transform=transform)
    elif config.dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                               download=True, transform=transform_test)
    elif config.dataset == "synthetic":
        # 兼容你的拼写；从 pt 里读 X,y 然后切分
        import math
        from torch.utils.data import TensorDataset, Subset

        path = "data/sythetic/dataset.pt"
        bundle = torch.load(path, map_location="cpu")
        X, y = bundle["X"], bundle["y"]
        n = len(y)

        # 80/20 切分（或读 config.train_ratio）
        train_ratio = getattr(config, "train_ratio", 0.8)
        n_train = int(math.floor(n * train_ratio))

        all_idx = np.arange(n)
        rng = np.random.default_rng(getattr(config, "seed", 0))
        rng.shuffle(all_idx)
        train_idx = all_idx[:n_train]
        test_idx  = all_idx[n_train:]

        ds_full = TensorDataset(X, y)
        trainset = Subset(ds_full, train_idx.tolist())
        testset  = Subset(ds_full, test_idx.tolist())
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    print(f"Loaded {len(trainset)} training samples and {len(testset)} test samples for {config.dataset}.")

    # Distribute training data among nodes
    all_train_indices = list(range(len(trainset)))
    np.random.shuffle(all_train_indices)
    samples_per_node = len(trainset) // config.num_nodes
    remaining_samples = len(trainset) % config.num_nodes
    current_pos = 0
    loader_batch_size = getattr(config, 'loader_batch_size', 64)

    trainloaders = []
    for i in range(config.num_nodes):
        node_subset_size = samples_per_node + (1 if i < remaining_samples else 0)
        node_indices = all_train_indices[current_pos: current_pos + node_subset_size]
        current_pos += node_subset_size
        node_subset = torch.utils.data.Subset(trainset, node_indices)
        trainloaders.append(torch.utils.data.DataLoader(
            node_subset,
            batch_size=loader_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=False
        ))

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    print(f"Data loading complete:")
    print(f"  - Training: {len(trainset)} samples distributed across {config.num_nodes} nodes")
    print(f"  - Test: {len(testset)} samples (reserved for final evaluation)")
    print_node_class_distribution(trainloaders)

    return trainloaders, testloader


from collections import defaultdict, Counter
from torch.utils.data import DataLoader, Subset

def load_extreme_non_iid(config):
    assert config.dataset == "mnist", "Extreme non-IID split currently only supports MNIST."

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=config.data_dir, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=config.data_dir, train=False,
                                         download=True, transform=transform)

    # Step 1: Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)

    # Step 2: Assign nodes to classes (each class gets N nodes)
    nodes_per_class = config.num_nodes // 10
    assert config.num_nodes % 10 == 0, "Number of nodes must be divisible by 10 for this split."

    trainloaders = []
    loader_batch_size = getattr(config, 'loader_batch_size', 64)

    for class_id in range(10):
        indices = class_to_indices[class_id]
        np.random.shuffle(indices)

        # Split into equal parts for nodes assigned to this class
        samples_per_node = len(indices) // nodes_per_class
        for i in range(nodes_per_class):
            start = i * samples_per_node
            end = (i + 1) * samples_per_node if i < nodes_per_class - 1 else len(indices)
            node_indices = indices[start:end]
            node_subset = Subset(trainset, node_indices)

            trainloaders.append(DataLoader(
                node_subset,
                batch_size=loader_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            ))

    testloader = DataLoader(
        testset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"Extreme non-IID MNIST data split complete:")
    print(f"  - Each of 10 classes assigned to {nodes_per_class} nodes")
    print(f"  - Total nodes: {len(trainloaders)}")
    print(f"  - Test set: {len(testset)} samples")

    print_node_class_distribution(trainloaders)

    return trainloaders, testloader

def print_node_class_distribution(trainloaders):
    print("\n📊 Per-Node Class Distribution:")
    for node_id, loader in enumerate(trainloaders):
        class_counter = Counter()
        for x, y in loader:
            class_counter.update(y.tolist())
        class_distribution = dict(sorted(class_counter.items()))
        print(f"  Node {node_id:02d}: {class_distribution}")


def load_moderate_non_iid(config):
    """
    Moderate non-IID split for MNIST:
    - 50 nodes
    - Each group of 10 nodes only sees 2 classes (0-1, 2-3, ..., 8-9)
    - Each node分到该组两类样本，混合shuffle后均匀分配
    """
    assert config.dataset == "mnist", "Only supports MNIST"
    assert config.num_nodes == 50, "This split assumes 50 nodes"

    from collections import defaultdict, Counter
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Step 1: Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root=config.data_dir, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=config.data_dir, train=False,
                                         download=True, transform=transform)

    # Step 2: Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)
    for k in class_to_indices:
        np.random.shuffle(class_to_indices[k])

    # Step 3: Assign node groups
    group_labels = [(i, i+1) for i in range(0, 10, 2)]  # [(0,1), (2,3), ...]
    nodes_per_group = config.num_nodes // len(group_labels)  # 10 nodes per group

    trainloaders = []
    loader_batch_size = getattr(config, 'loader_batch_size', 64)

    for group_id, (label1, label2) in enumerate(group_labels):
        # 合并该组的两个label的样本index
        indices = class_to_indices[label1] + class_to_indices[label2]
        np.random.shuffle(indices)  # 充分混合

        samples_per_node = len(indices) // nodes_per_group
        for i in range(nodes_per_group):
            start = i * samples_per_node
            end = (i + 1) * samples_per_node if i < nodes_per_group - 1 else len(indices)
            node_indices = indices[start:end]
            node_subset = Subset(trainset, node_indices)
            trainloaders.append(DataLoader(
                node_subset,
                batch_size=loader_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            ))

    testloader = DataLoader(
        testset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"Moderate non-IID MNIST split (2 classes per 10 nodes) complete:")
    print(f"  - Each group of 10 nodes sees only classes: {group_labels}")
    print(f"  - Test set: {len(testset)} samples")

    print_node_class_distribution(trainloaders)
    return trainloaders, testloader


def load_moderate_non_iid_cifar10(config):
    """
    Moderate non-IID split for CIFAR-10, 30 nodes:
    - 5 groups, each group of 6 nodes only sees 2 classes (0-1, 2-3, ..., 8-9)
    - 每节点拿到组内2类混合且平均分配的样本
    """
    assert config.dataset == "cifar10"
    assert config.num_nodes == 30, "This split assumes 30 nodes"
    import numpy as np
    from collections import defaultdict, Counter
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms

    # Step 1: Load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                           download=True, transform=transform_test)

    # Step 2: Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)
    for k in class_to_indices:
        np.random.shuffle(class_to_indices[k])

    # Step 3: Assign node groups
    group_labels = [(i, i+1) for i in range(0, 10, 2)]  # [(0,1), (2,3), (4,5), (6,7), (8,9)]
    nodes_per_group = config.num_nodes // len(group_labels)  # 6 nodes per group

    trainloaders = []
    loader_batch_size = getattr(config, 'loader_batch_size', 64)

    for group_id, (label1, label2) in enumerate(group_labels):
        # 合并该组的两个label的样本index
        indices = class_to_indices[label1] + class_to_indices[label2]
        np.random.shuffle(indices)  # 充分混合

        samples_per_node = len(indices) // nodes_per_group
        for i in range(nodes_per_group):
            start = i * samples_per_node
            end = (i + 1) * samples_per_node if i < nodes_per_group - 1 else len(indices)
            node_indices = indices[start:end]
            node_subset = Subset(trainset, node_indices)
            trainloaders.append(DataLoader(
                node_subset,
                batch_size=loader_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            ))

    testloader = DataLoader(
        testset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"Moderate non-IID CIFAR-10 split (2 classes per 6 nodes, total 30 nodes) complete:")
    print(f"  - Each group of 6 nodes sees only classes: {group_labels}")
    print(f"  - Test set: {len(testset)} samples")
    print_node_class_distribution(trainloaders)
    return trainloaders, testloader


def load_extreme_non_iid_cifar10(config):
    """
    Extreme non-IID split for CIFAR-10, 30 nodes:
    - Each node only sees samples from a single class
    - Total 30 nodes, each node gets samples from one of the 10 classes
    """
    assert config.dataset == "cifar10"
    assert config.num_nodes == 30, "This split assumes 30 nodes"

    import numpy as np
    from collections import defaultdict, Counter
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms

    # Step 1: Load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=config.data_dir, train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False,
                                           download=True, transform=transform_test)

    # Step 2: Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_to_indices[label].append(idx)
    
    for k in class_to_indices:
        np.random.shuffle(class_to_indices[k])

    # Step 3: Assign nodes to classes (each node gets samples from one class)
    nodes_per_class = config.num_nodes // 10
    assert config.num_nodes % 10 == 0, "Number of nodes must be divisible by 10 for this split."

    trainloaders = []
    loader_batch_size = getattr(config, 'loader_batch_size', 64)

    for class_id in range(10):
        indices = class_to_indices[class_id]
        np.random.shuffle(indices)

        samples_per_node = len(indices) // nodes_per_class
        for i in range(nodes_per_class):
            start = i * samples_per_node
            end = (i + 1) * samples_per_node if i < nodes_per_class - 1 else len(indices)
            node_indices = indices[start:end]
            node_subset = Subset(trainset, node_indices)
            trainloaders.append(DataLoader(
                node_subset,
                batch_size=loader_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            ))      
    testloader = DataLoader(
        testset,
        batch_size=loader_batch_size,   
        shuffle=False,
        num_workers=0,
        pin_memory=False,   
        drop_last=False
    )
    print(f"Extreme non-IID CIFAR-10 split complete:")
    print(f"  - Each of 10 classes assigned to {nodes_per_class} nodes")
    print(f"  - Total nodes: {len(trainloaders)}")
    print(f"  - Test set: {len(testset)} samples")  
    print_node_class_distribution(trainloaders)
    return trainloaders, testloader
