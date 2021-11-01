import numpy as np
from sklearn import linear_model
def elements(array):
    return array.ndim and array.size
def sigmoid(x):
    return 1/(1+np.exp(-x))
def generate_dataset(dim=5,max_arm=100):
    theta_star = np.random.multivariate_normal(np.zeros(dim),3/(dim**3)*np.eye(dim))
    X = np.ones([max_arm, dim])
    for i in range(max_arm):
        xi = np.random.uniform(-1,1,dim)
        X[i,:] = xi/np.linalg.norm(xi)
    # mean = sigmoid(X @ theta_star)
    # for j in range(max_arm):
    #     Y[:,j] = np.random.binomial(1, mean[j], prob_ins)+np.random.normal(0,noise_sigma,prob_ins)
    return (X,theta_star)
def solve_MLE(X,y,lam=1):
    model = linear_model.LogisticRegression(C=lam,fit_intercept=False,solver='newton-cg')
    model.fit(X,y)
    return model.coef_
def GLM_TSL_new(dim=5,max_arm=100,horizon=50000, noise_sigma=1,lam=1):
    (X,theta_star) = generate_dataset(dim, max_arm)
    p = sigmoid(X @ theta_star)
    while(1/(np.min(p*(1-p))+0.00001) > horizon**(1/5)):
        (X, theta_star) = generate_dataset(dim, max_arm)
        p = sigmoid(X @ theta_star)
    print(f'dataset generated! kappa={1/np.min(p*(1-p)+0.00001)}')
    theta_t = np.random.normal(0,1,X.shape[1]) # init \bar\theta_t
    mu1 = np.max(sigmoid(X@theta_star)) # optimal arm
    reward = []
    hessian = np.eye(X.shape[1])
    arm_pulled = np.array([])
    sum = 0
    # constants
    S = np.linalg.norm(theta_star)
    gamma = np.sqrt(2*lam) * (S+1/2)+2/np.sqrt(2*lam)*np.log(2**dim/(1/horizon)*(1+1/4 * 1/(dim*2*lam))**(dim/2))
    c1 = (1+2*S)**(-1) * gamma
    regret = []
    while len(np.unique(reward)) < 2: # warm up till 2 classes for sklearn to solve
        theta_tilde = np.random.multivariate_normal(theta_t, (c1 ** 2) * np.linalg.inv(hessian))  # optimistic
        arm_idx = np.argmax(X @ theta_tilde)  # make decision
        if elements(arm_pulled) == 0:
            hessian = 2 * lam * np.eye(len(X[arm_idx]))
            arm_pulled = X[arm_idx].reshape(1, -1)
        else:
            arm_pulled = np.concatenate([arm_pulled, X[arm_idx].reshape(1, -1)], axis=0)
        mu = sigmoid(theta_t.reshape(1, -1) @ X[arm_idx].reshape(-1, 1))
        hessian = hessian + (X[arm_idx].reshape(-1, 1) @ X[arm_idx].reshape(1, -1)) * mu * (1 - mu)
        imm_reward = np.random.binomial(1, p[arm_idx], 1)
        reward.append(imm_reward.item())
    for step in range(horizon):
        theta_tilde = np.random.multivariate_normal(theta_t, (c1**2)*np.linalg.inv(hessian)) # optimistic
        arm_idx = np.argmax(X@theta_tilde) # make decision
        arm_pulled = np.concatenate([arm_pulled,X[arm_idx].reshape(1,-1)],axis=0)
        imm_reward = np.random.binomial(1, p[arm_idx],1)
        reward.append(imm_reward.item())
        sum += mu1 - p[arm_idx] # cumulative regret
        regret.append(sum)
        theta_t = solve_MLE(arm_pulled, np.array(reward), 1/lam).reshape(-1)
        # S = np.linalg.norm(theta_star)
        gamma = np.sqrt(2*lam) * (S + 1 / 2) + 2 / np.sqrt(2*lam) * np.log(
            2 ** dim / (1 / horizon) * (1 + 1 / 4 * (step+1) / (dim * 2*lam)) ** (dim / 2))
        c1 = (1 + 2*S) ** (-1) * gamma
        mu = sigmoid(theta_t.reshape(1,-1) @ X[arm_idx].reshape(-1,1))
        hessian = hessian + (X[arm_idx].reshape(-1,1) @ X[arm_idx].reshape(1,-1))* mu * (1-mu)
        if step % (0.1*horizon) == 0:
            print(f'step {step}!')
    return regret
