import numpy as np 

def approval_voting(V, S, k):
    S_array = np.array(S)
    n_projects = len(S)
    n_voters = len(V)
    
    approval_matrix = np.zeros((n_voters, n_projects), dtype=bool)
    
    for i, votes in enumerate(V):
        for project in votes:
            if project in S:
                project_idx = np.where(S_array == project)[0][0]
                approval_matrix[i, project_idx] = True
    
    approval_counts = np.sum(approval_matrix, axis=0)
    top_k_indices = np.argsort(approval_counts)[::-1][:k]
    selected = [S_array[i] for i in top_k_indices]
    
    return selected

def proportional_approval_voting(V, S, k):
    S_array = np.array(S)
    n_projects = len(S)
    n_voters = len(V)
    
    approval_matrix = np.zeros((n_voters, n_projects), dtype=bool)
    for i, votes in enumerate(V):
        for project in votes:
            if project in S:
                project_idx = np.where(S_array == project)[0][0]
                approval_matrix[i, project_idx] = True
    
    selected_indices = []
    remaining_indices = np.arange(n_projects)
    
    for _ in range(k):
        if len(remaining_indices) == 0:
            break
            
        best_idx = None
        best_marginal_score = -np.inf
        
        for idx in remaining_indices:
            marginal_score = calculate_marginal_pav_score_numpy(
                approval_matrix, selected_indices, idx
            )
            
            if marginal_score > best_marginal_score:
                best_marginal_score = marginal_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices = remaining_indices[remaining_indices != best_idx]
    
    selected = [S_array[i] for i in selected_indices]
    return selected

def calculate_marginal_pav_score_numpy(approval_matrix, current_selected, new_project_idx):
    voters_approving_new = approval_matrix[:, new_project_idx]
    
    if not np.any(voters_approving_new):
        return 0.0
    
    if len(current_selected) == 0:
        return np.sum(voters_approving_new)
    
    current_approved_counts = np.sum(approval_matrix[:, current_selected], axis=1)
    relevant_counts = current_approved_counts[voters_approving_new]
    marginal_contributions = 1.0 / (relevant_counts + 1)
    return np.sum(marginal_contributions)

def greedy_voting(V, S, k):
    S_array = np.array(S)
    n_projects = len(S)
    n_voters = len(V)
    
    approval_matrix = np.zeros((n_voters, n_projects), dtype=bool)
    for i, votes in enumerate(V):
        for project in votes:
            if project in S:
                project_idx = np.where(S_array == project)[0][0]
                approval_matrix[i, project_idx] = True
    
    selected_indices = []
    remaining_indices = np.arange(n_projects)
    satisfied_voters = np.zeros(n_voters, dtype=bool)
    
    for _ in range(k):
        if len(remaining_indices) == 0:
            break
        
        unsatisfied_voters = ~satisfied_voters
        if not np.any(unsatisfied_voters):
            break
            
        project_satisfaction_counts = np.sum(
            approval_matrix[unsatisfied_voters][:, remaining_indices], 
            axis=0
        )
        
        if np.max(project_satisfaction_counts) == 0:
            break
        
        best_relative_idx = np.argmax(project_satisfaction_counts)
        best_idx = remaining_indices[best_relative_idx]
        selected_indices.append(best_idx)
        satisfied_voters |= approval_matrix[:, best_idx]
        remaining_indices = remaining_indices[remaining_indices != best_idx]
    
    selected = [S_array[i] for i in selected_indices]
    return selected

projects = ['A', 'B', 'C', 'D', 'E']
voters = [
    ['A', 'B'],
    ['A', 'C'], 
    ['A', 'E'],
    ['C', 'D', 'E'],
    ['A', 'E']
]
k = 3
    
print("Projects:", projects)
print("Voters:", voters)
print("k =", k)
print()

av_result = approval_voting(voters, projects, k)
print("Approval Voting result:", av_result)

pav_result = proportional_approval_voting(voters, projects, k)
print("Proportional Approval Voting result:", pav_result)

greedy_result = greedy_voting(voters, projects, k)
print("Greedy Voting result:", greedy_result)
    