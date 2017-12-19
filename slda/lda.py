import numpy as np
from scipy.special import digamma


class SLDA():
    '''Latent Dirichlet Allocation.
    Stochastic Variational Inference of LDA.

    Parameters
    ----------
    num_topic : int, number of topics.
    vocab_size : int, number of words in vocabulary.
    alpha : float, prior parameter of global variable (default = 1 / num_topic).
    eta : float, prior parameter of local variable (default = 1 / num_topic).
    kappa : float, learning decay rate between (0.5, 1.0] (default = 0.6).
    tolerance : float, threshold for convergence check (default = 1e-3).
    outer_iter_max : int, number of epochs (default = 10).
    inner_iter_max : int, number of iterations for local variable (default = 100).
    verbose : bool, if True, print detailed log (default = False).

    Attributes
    ----------
    fitted : bool, flag to check if the model has been fitted.
    lambda_ : float matrix of shape (vocab_size, num_topic), word portions of each topic.
    gamma : float matrix of shape (num_document, num_topic), local Dirichlet parameters.
    theta : float matrix of shape (num_document, num_topic), topic portions.
    '''
    def __init__(self, num_topic, vocab_size, alpha=None, eta=None, kappa=0.6, tolerance=1e-3, outer_iter_max=10, inner_iter_max=100, verbose=False):
        self.K = num_topic
        self.V = vocab_size
        self.tolerance = tolerance
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 1 / self.K
        if eta is not None:
            self.eta = eta
        else:
            self.eta = 1 / self.K
        self.kappa = kappa
        self.outer_iter_max = outer_iter_max
        self.inner_iter_max = inner_iter_max
        self.verbose = verbose

        self.fitted = False
        self.vocab = np.eye(self.V, dtype=int)

    def compute_local_var(self, w_d):
        '''helper function for local iterations'''
        N = len(w_d)
        gamma_d = np.ones(self.K)
        last_gamma_d = np.ones(self.K)
        for j in range(self.inner_iter_max):
            digamma_gamma_d = digamma(gamma_d)
            E = np.tile(digamma_gamma_d - self.sum_digamma_lambda, (N, 1)).T
            E += self.digamma_lambda[w_d, :].T
            E -= E.max(axis=0)

            psi = np.exp(E)
            psi /= psi.sum(axis=0)
            gamma_d = self.alpha + psi.sum(axis=1)

            if abs(gamma_d - last_gamma_d).mean() < self.tolerance:
                break
                
            last_gamma_d = gamma_d

        return gamma_d, psi

    def fit(self, X, y=None, X_test=None):
        '''fit model with data.

        Parameters
        ----------
        X : dataset for training.
        y : labels corresponding to data points in X. If provided, evaluate model
            in terms of accuracy.
        X_test : dataset for testing. If provideds, evaluate model in terms of
            log predictive probability.
        '''

        data = X
        self.D = len(data)
        self.lambda_ = np.random.exponential(self.K*self.V/(300*self.D), (self.V, self.K)) + self.eta
        self.theta = np.ones([self.V, self.K]) / self.V
        self.gamma = np.zeros([self.D, self.K])
        last_lambda = self.lambda_
        self.digamma_lambda = digamma(self.lambda_)
        self.sum_digamma_lambda = digamma(np.sum(self.lambda_, axis=0))
        t = 0
        if X_test is not None:
            self.fitted = True
            lpp = self.lpp(X_test)
            self.fitted = False
            lpp_list = [lpp]
            print('init log predictive probability:', lpp)

        if y is not None:
            self.fitted = True
            preds = self.predict(X)
            acc = self.acc(preds, y)[0]
            self.fitted = False
            acc_list = [acc]
            print('init accuracy (under unsupervised training):', acc)

        for i in range(self.outer_iter_max):
            for w_d, d in data:
                w_d, d = w_d.numpy()[0][:-1], d.numpy()[0]
                N = len(w_d)

                psi = np.zeros([self.K, N])
                
                self.digamma_lambda = digamma(self.lambda_)
                self.sum_digamma_lambda = digamma(np.sum(self.lambda_, axis=0))

                self.gamma[d], psi = self.compute_local_var(w_d)

                lambda_hat = np.ones([self.V, self.K]) * self.eta
                
                for k in range(self.K):
                    for n in range(N):
                        word_vec = self.vocab[w_d[n]]
                        lambda_hat[:, k] += self.D * psi[k, n] * word_vec

                pho = (t+1) ** (-self.kappa)
                t += 1

                self.lambda_ = (1 - pho) * self.lambda_ + pho * lambda_hat

            if self.verbose:
                    print('epoch', i+1, 'parameters average change:', abs(self.lambda_ - last_lambda).mean())

            if X_test is not None:
                self.fitted = True
                self.theta = self.lambda_ / self.lambda_.sum(axis=0)
                lpp = self.lpp(X_test)
                print('log predictive probability:', lpp)
                lpp_list.append(lpp)
                self.fitted = False

            if y is not None:
                self.fitted = True
                preds = self.predict(X)
                acc = self.acc(preds, y)[0]
                print('accuracy (under unsupervised training):', acc)
                acc_list.append(acc)
                self.fitted = False

            if abs(self.lambda_ - last_lambda).mean() < self.tolerance * 100:
                print('converged at', i+1)
                break
            
            last_lambda = self.lambda_
        self.theta = self.lambda_ / self.lambda_.sum(axis=0)
        self.fitted = True

        if X_test is not None and y is not None:
            return lpp_list, acc_list
        elif X_test is not None:
            return lpp_list
        elif y is not None:
            return acc_list

    def predict(self, X, return_label=True):
        '''predict result with fitted parameters.

        Parameters
        ----------
        X : dataset for prediction.
        return_label : If True, return a single index of most possible topic.
            Otherwise, return the topic portions.
        '''
        if not self.fitted:
            raise ValueError('You must fit the model first.')
        
        if return_label:
            return np.array([np.argmax(self.compute_local_var(w_d)[0]) for w_d in X])
        else:
            gamma_test = np.array([self.compute_local_var(w_d)[0] for w_d in X])
            topic_dist = (gamma_test.T / gamma_test.sum(axis=1)).T
            return topic_dist

    def acc(self, preds, labels):
        '''compute accuracy.'''
        if not self.fitted:
            raise ValueError('You must fit the model first.')
            
        D = len(labels)
        topic = [np.argmax(np.histogram(np.extract([labels == i], preds), bins=self.K, range=(0, self.K-1))[0]) for i in range(self.K)]
        if len(set(topic)) != self.K:
            return 0, 0
        else:
            correct = np.array([sum(np.extract([labels == i], preds) == topic[i]) for i in range(self.K)])
            total = np.array([sum(labels == i) for i in range(self.K)])
            return sum(correct) / sum(total), correct / total

    def lpp(self, X_test):
        '''compute log predictive probablity.'''
        if not self.fitted:
            raise ValueError('You must fit the model first.')

        topic_dist = self.predict(X_test, return_label=False)  # D * K
        total_words, total_log_prob = 0, 0
        for i, w_d in enumerate(X_test):
            topic_dist_d = topic_dist[i:i+1]  # 1 * K
            theta_tmp = self.theta[w_d]  # N * K
            total_log_prob += np.log(np.dot(theta_tmp, topic_dist_d.T)).sum()  # N * 1
            total_words += len(w_d)

        return total_log_prob / total_words

    def top_words(self, topic_idx, idx_to_word, num_words=10):
        top_n = sorted(self.theta[:, topic_idx], reverse=True)[num_words]
        idx = [i for i, value in enumerate(self.theta[:, topic_idx]) if value > top_n]
        for i in idx:
            print(idx_to_word[i], self.theta[i, topic_idx])
