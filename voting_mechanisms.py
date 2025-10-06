from collections import defaultdict

def approval_voting(V, S, k):
    approval_counts = defaultdict(int)
    for votes in V:
        for project in votes:
            if project in S:    # shouldn't have any non S elements, but just in case
                approval_counts[project] += 1
    
    sorted_projects = sorted(approval_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [project for project, count in sorted_projects[:k]]
    return selected

def proportional_approval_voting(V, S, k):
    selected = []
    remaining_projects = set(S)
    
    for _ in range(k):
        if not remaining_projects:
            break
            
        best_project = None
        best_marginal_score = -float('inf')
        
        for project in remaining_projects:
            marginal_score = calculate_marginal_pav_score(V, selected, project)
            
            if marginal_score > best_marginal_score:
                best_marginal_score = marginal_score
                best_project = project
        
        if best_project is not None:
            selected.append(best_project)
            remaining_projects.remove(best_project)
    
    return selected

def calculate_marginal_pav_score(V, current_selected, new_project):
    marginal_score = 0
    current_set = set(current_selected)
    
    for votes in V:
        votes_set = set(votes)
        
        if new_project in votes_set:
            current_approved = len(votes_set.intersection(current_set))
            marginal_score += 1.0 / (current_approved + 1)
    
    return marginal_score

def greedy_voting(V, S, k):
    selected = []
    remaining_votes = V.copy()
    remaining_projects = set(S)
    
    for _ in range(k):
        if not remaining_projects:
            break

        counts = defaultdict(int)
        for votes in remaining_votes:
            for project in votes:
                if project in remaining_projects:
                    counts[project] += 1
        
        if not counts:
            break
            
        sorted_projects = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        best_project = sorted_projects[0][0]  # Get the project name, not a list
        
        selected.append(best_project)
        remaining_projects.remove(best_project)
        remaining_votes = [votes for votes in remaining_votes if best_project not in votes]
    
    return selected

# Example
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