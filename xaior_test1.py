"""
Toygarhan Tug
A Draft for a simple test code for:
Complete Lot Sizing with Carbon Emissions + XAI Implementation
Based on Koca & Koksalan (2023) - Capacitated lot sizing with periodic carbon emission constraints
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_random_parameters():
    """Generate random problem parameters for lot sizing with carbon constraints"""
    
    params = {
        'T': 12,  # 12 time periods
        'M': 2,   # 2 resources (1 green, 1 regular)
        'demand': np.random.randint(50, 200, 12),  # Random demand per period
        'setup_costs': [np.random.uniform(100, 300), np.random.uniform(200, 500)],  # f1, f2
        'unit_costs': [np.random.uniform(10, 30), np.random.uniform(15, 35)],       # p1, p2
        'capacities': [np.random.randint(120, 180), np.random.randint(100, 150)],   # C1, C2
        'holding_cost': np.random.uniform(1, 3),  # Inventory holding cost per unit
        'emission_coeffs': [0.5, 1.8],  # e1 (green), e2 (regular) - fixed for consistency
        'max_emission_avg': np.random.uniform(0.9, 1.3)  # Maximum average emission per unit
    }
    return params

def solve_capacitated_lot_sizing_carbon(params):
    """
    Solve the capacitated lot sizing problem with periodic carbon emission constraints
    Based on the mathematical model from the paper of Koca & Koksalan (2023)
    """
    try:
        # Create model
        model = gp.Model("CLS_PC")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output
        model.setParam('TimeLimit', 30)  # 30 second time limit
        
        T = params['T']
        M = params['M']
        
        # Decision variables
        x = model.addVars(T, M, name="production", lb=0)  # Production quantities
        y = model.addVars(T, M, vtype=GRB.BINARY, name="setup")  # Setup decisions
        s = model.addVars(T, name="inventory", lb=0)  # Inventory levels
        
        # Objective: minimize total cost (setup + production + holding)
        obj = (gp.quicksum(params['setup_costs'][m] * y[t,m] + 
                          params['unit_costs'][m] * x[t,m] 
                          for t in range(T) for m in range(M)) +
               gp.quicksum(params['holding_cost'] * s[t] for t in range(T)))
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        
        # 1. Flow balance constraints
        for t in range(T):
            if t == 0:
                model.addConstr(
                    gp.quicksum(x[t,m] for m in range(M)) == 
                    params['demand'][t] + s[t],
                    name=f"balance_{t}"
                )
            else:
                model.addConstr(
                    s[t-1] + gp.quicksum(x[t,m] for m in range(M)) == 
                    params['demand'][t] + s[t],
                    name=f"balance_{t}"
                )
        
        # 2. Capacity constraints (linking x and y)
        for t in range(T):
            for m in range(M):
                model.addConstr(
                    x[t,m] <= params['capacities'][m] * y[t,m],
                    name=f"capacity_{t}_{m}"
                )
        
        # 3. Periodic carbon emission constraints
        # Average emission per unit produced should not exceed max_emission_avg
        for t in range(T):
            total_production = gp.quicksum(x[t,m] for m in range(M))
            total_emissions = gp.quicksum(params['emission_coeffs'][m] * x[t,m] for m in range(M))
            
            # Only add constraint if there's production
            model.addGenConstrIndicator(
                model.addVar(vtype=GRB.BINARY), True,
                total_emissions <= params['max_emission_avg'] * total_production,
                name=f"emission_{t}"
            )
            
            # Alternative simpler constraint (always active)
            model.addConstr(
                total_emissions <= params['max_emission_avg'] * (total_production + 0.01),
                name=f"emission_simple_{t}"
            )
        
        # Solve the model
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Extract solution
            production_sol = [[x[t,m].X for m in range(M)] for t in range(T)]
            setup_sol = [[int(y[t,m].X) for m in range(M)] for t in range(T)]
            inventory_sol = [s[t].X for t in range(T)]
            
            # Calculate metrics
            total_cost = model.objVal
            total_production = sum(sum(production_sol[t]) for t in range(T))
            total_emissions = sum(params['emission_coeffs'][m] * production_sol[t][m] 
                                for t in range(T) for m in range(M))
            avg_emission = total_emissions / max(total_production, 1)
            
            # Resource utilization
            resource1_usage = sum(production_sol[t][0] for t in range(T))
            resource2_usage = sum(production_sol[t][1] for t in range(T))
            resource1_ratio = resource1_usage / max(total_production, 1)
            
            solution = {
                'production': production_sol,
                'setup': setup_sol,
                'inventory': inventory_sol,
                'total_cost': total_cost,
                'total_emissions': total_emissions,
                'avg_emission': avg_emission,
                'total_production': total_production,
                'resource1_usage': resource1_usage,
                'resource2_usage': resource2_usage,
                'resource1_ratio': resource1_ratio,
                'feasible': True
            }
            
        else:
            solution = {'feasible': False}
            
        model.dispose()
        return solution
        
    except Exception as e:
        print(f"Error solving optimization: {e}")
        return {'feasible': False}

def generate_optimization_dataset(n_instances=1000):
    """Generate dataset by solving multiple optimization instances. I just generate random instances here, didn't take any existing dataset"""
    print(f"Generating {n_instances} optimization instances...")
    
    results = []
    failed_count = 0
    
    for i in range(n_instances):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_instances} instances...")
            
        # Generate random parameters
        params = generate_random_parameters()
        
        # Solve optimization problem
        solution = solve_capacitated_lot_sizing_carbon(params)
        
        # Keep only feasible solutions
        if solution.get('feasible', False):
            results.append({
                'params': params,
                'solution': solution
            })
        else:
            failed_count += 1
    
    print(f"Generated {len(results)} feasible instances ({failed_count} failed)")
    return results

def convert_to_ml_dataset(optimization_results):
    """Convert optimization results to ML-ready format"""
    print("Converting optimization results to ML dataset...")
    
    features = []
    targets = []
    
    for instance in optimization_results:
        params = instance['params']
        solution = instance['solution']
        
        # INPUT FEATURES (X) - Problem parameters
        feature_row = []
        
        # Demand pattern (12 values)
        feature_row.extend(params['demand'])
        
        # Cost parameters (4 values)
        feature_row.extend(params['setup_costs'])
        feature_row.extend(params['unit_costs'])
        
        # Resource parameters (2 values)
        feature_row.extend(params['capacities'])
        
        # Emission parameters (1 value)
        feature_row.append(params['max_emission_avg'])
        
        # Holding cost (1 value)
        feature_row.append(params['holding_cost'])
        
        # Derived features
        feature_row.append(np.mean(params['demand']))  # Average demand
        feature_row.append(np.std(params['demand']))   # Demand variability
        feature_row.append(sum(params['demand']))      # Total demand
        feature_row.append(params['setup_costs'][1] - params['setup_costs'][0])  # Setup cost diff
        feature_row.append(params['unit_costs'][1] - params['unit_costs'][0])    # Unit cost diff
        
        # OUTPUT TARGETS (y) - Optimization decisions and outcomes
        target_row = [
            solution['total_cost'],
            solution['total_emissions'],
            solution['avg_emission'],
            solution['resource1_ratio'],
            solution['total_production']
        ]
        
        features.append(feature_row)
        targets.append(target_row)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(targets)
    
    print(f"ML dataset shape: {X.shape} features, {y.shape} targets")
    return X, y

def create_feature_names():
    """Creating meaningful names for all features here so that we can read them at the end"""
    names = []
    
    # Demand features
    names.extend([f'demand_period_{t+1}' for t in range(12)])
    
    # Cost features
    names.extend(['setup_cost_green', 'setup_cost_regular'])
    names.extend(['unit_cost_green', 'unit_cost_regular'])
    
    # Capacity features
    names.extend(['capacity_green', 'capacity_regular'])
    
    # Emission features
    names.append('max_emission_avg')
    
    # Other features
    names.append('holding_cost')
    
    # Derived features
    names.extend(['avg_demand', 'demand_std', 'total_demand', 
                  'setup_cost_diff', 'unit_cost_diff'])
    
    return names

def create_target_names():
    """Create names for target variables"""
    return ['total_cost', 'total_emissions', 'avg_emission', 
            'green_resource_ratio', 'total_production']

def train_ml_models(X, y, feature_names, target_names):
    """Train multiple ML models for different prediction tasks"""
    print("Training ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {}
    performance = {}
    
    # Train separate models for each target
    for i, target_name in enumerate(target_names):
        print(f"  Training model for {target_name}...")
        
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train[:, i])
        
        # Evaluate performance
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train[:, i], y_pred_train)
        test_r2 = r2_score(y_test[:, i], y_pred_test)
        
        models[target_name] = model
        performance[target_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"    {target_name}: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    
    return models, (X_train, X_test, y_train, y_test), performance

def generate_shap_explanations(models, X_test, feature_names, target_names):
    """Generate SHAP explanations for all models"""
    print("Generating SHAP explanations...")
    
    explanations = {}
    
    # Focus on most important models for carbon emissions
    key_models = ['total_emissions', 'avg_emission', 'green_resource_ratio']
    
    for target_name in key_models:
        if target_name in models:
            print(f"  Generating explanations for {target_name}...")
            
            model = models[target_name]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use subset for speed)
            sample_size = min(200, len(X_test))
            X_sample = X_test[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            explanations[target_name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_sample': X_sample,
                'expected_value': explainer.expected_value
            }
    
    return explanations

def create_visualizations(explanations, feature_names, target_names):
    """Create SHAP visualizations and insights"""
    print("Creating visualizations and insights...")
    
    # Create summary plots for key models
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    plot_idx = 0
    for target_name, exp_data in explanations.items():
        if plot_idx < 4:
            plt.subplot(2, 2, plot_idx + 1)
            shap.summary_plot(
                exp_data['shap_values'], 
                exp_data['X_sample'],
                feature_names=feature_names,
                show=False,
                max_display=10
            )
            plt.title(f'SHAP Summary: {target_name}')
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('shap_summary_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate feature importance insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM XAI ANALYSIS")
    print("="*80)
    
    for target_name, exp_data in explanations.items():
        print(f"\n{target_name.upper()} - Key Drivers:")
        print("-" * 50)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(exp_data['shap_values']), axis=0)
        
        # Get top 10 most important features
        top_indices = np.argsort(mean_shap)[-10:][::-1]
        
        for i, idx in enumerate(top_indices[:5]):
            feature_name = feature_names[idx]
            importance = mean_shap[idx]
            print(f"  {i+1}. {feature_name}: {importance:.3f}")
        
        # Calculate directional effects
        positive_effect = np.mean(exp_data['shap_values'][exp_data['shap_values'] > 0])
        negative_effect = np.mean(exp_data['shap_values'][exp_data['shap_values'] < 0])
        
        print(f"  Average positive impact: +{positive_effect:.3f}")
        print(f"  Average negative impact: {negative_effect:.3f}")

def analyze_carbon_emission_patterns(explanations, feature_names):
    """Analyze patterns in carbon emission decisions"""
    print("\n" + "="*80)
    print("CARBON EMISSION PATTERN ANALYSIS")
    print("="*80)
    
    if 'total_emissions' in explanations:
        shap_values = explanations['total_emissions']['shap_values']
        X_sample = explanations['total_emissions']['X_sample']
        
        # Find demand-related features
        demand_features = [i for i, name in enumerate(feature_names) if 'demand' in name.lower()]
        
        print("\nDemand Impact on Emissions:")
        for feat_idx in demand_features[:6]:  # First 6 periods
            feat_name = feature_names[feat_idx]
            avg_impact = np.mean(shap_values[:, feat_idx])
            print(f"  {feat_name}: {avg_impact:+.2f}")
        
        # Analyze emission limit impact
        emission_limit_idx = feature_names.index('max_emission_avg')
        emission_impact = shap_values[:, emission_limit_idx]
        
        print(f"\nEmission Limit Impact:")
        print(f"  Average impact: {np.mean(emission_impact):+.3f}")
        print(f"  When strict limits (low values): {np.mean(emission_impact[X_sample[:, emission_limit_idx] < 1.0]):+.3f}")
        print(f"  When loose limits (high values): {np.mean(emission_impact[X_sample[:, emission_limit_idx] > 1.2]):+.3f}")
    
    if 'green_resource_ratio' in explanations:
        print(f"\nGreen Resource Usage Patterns:")
        shap_values = explanations['green_resource_ratio']['shap_values']
        X_sample = explanations['green_resource_ratio']['X_sample']
        
        # Cost difference impact
        cost_diff_idx = feature_names.index('setup_cost_diff')
        cost_impact = shap_values[:, cost_diff_idx]
        print(f"  Setup cost difference impact: {np.mean(cost_impact):+.3f}")
        print(f"  (Positive = more green resource usage when regular resource is expensive)")

def main():
    """Main execution function"""
    print("CAPACITATED LOT SIZING WITH CARBON CONSTRAINTS + XAI")
    print("=" * 60)
    
    # Step 1: Generate optimization dataset
    print("\nStep 1: Generating optimization dataset...")
    optimization_results = generate_optimization_dataset(n_instances=800)
    
    if len(optimization_results) < 100:
        print("Error: Not enough feasible solutions generated!")
        return
    
    # Step 2: Convert to ML format
    print("\nStep 2: Converting to ML format...")
    X, y = convert_to_ml_dataset(optimization_results)
    feature_names = create_feature_names()
    target_names = create_target_names()
    
    # Step 3: Train ML models
    print("\nStep 3: Training ML models...")
    models, data_splits, performance = train_ml_models(X, y, feature_names, target_names)
    
    X_train, X_test, y_train, y_test = data_splits
    
    # Step 4: Generate SHAP explanations
    print("\nStep 4: Generating SHAP explanations...")
    explanations = generate_shap_explanations(models, X_test, feature_names, target_names)
    
    # Step 5: Create visualizations and insights
    print("\nStep 5: Creating visualizations and insights...")
    create_visualizations(explanations, feature_names, target_names)
    
    # Step 6: Analyze carbon emission patterns
    analyze_carbon_emission_patterns(explanations, feature_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Key outputs:")
    print("- SHAP summary plots saved as 'shap_summary_plots.png'")
    print("- ML model performance metrics displayed above")
    print("- Carbon emission insights provided above")
    print("\nThis analysis reveals how different factors influence carbon emissions")
    print("in lot sizing decisions, providing explainable insights for sustainable operations.")

if __name__ == "__main__":
    main()