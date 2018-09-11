
# coding: utf-8

# ### pbt: Population Based Training
# 
# Code to replicate figure 2 of [Population Based Training of Neural Networks, Jaderberg et al](https://arxiv.org/abs/1711.09846)
# 

# In[ ]:


from __future__ import print_function
import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)

_ = np.random.seed(123)


# In[ ]:


# Define PBT worker

class Worker():
    
    def __init__(self, theta, h, objective, surrogate_objective, id):

        self.id = id
        
        self._history = {"theta" : [], "h" : [], "score" : []}
        
        with tf.name_scope("worker_" + str(self.id)):
            self.h_ = tf.placeholder(tf.float32, shape=np.shape(h), name="input_h")
            self.theta_ = tf.placeholder(tf.float32, shape=np.shape(theta), name="input_theta")
            self.h = tf.Variable(h, name="hyperparam", dtype=tf.float32)
            self.theta = tf.Variable(theta, name="theta", dtype=tf.float32)
            
            self.h_assign = tf.assign(self.h, self.h_)
            self.theta_assign = tf.assign(self.theta, self.theta_)
        
        self.objective = objective(self.theta)
        self.surrogate_objective = surrogate_objective(self.theta, self.h)
        self.loss, self.update_step = self._init_graph()
    
    def _init_graph(self):
        loss = -1 * self.surrogate_objective
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
        update_step = optimizer.minimize(loss)
        return loss, update_step
    
    @property
    def history(self):
        return {
            "theta" : np.vstack(self._history['theta']),
            "h"     : np.vstack(self._history['h']),
            "score" : np.array(self._history['score']),
        }
    
    def _log(self, sess):
        theta, h = sess.run([self.theta, self.h])
        self._history['theta'].append(np.copy(theta))
        self._history['h'].append(np.copy(h))
        self._history['score'].append(self.eval(sess))
    
    
    def step(self, sess):
        """ Take an optimization step, given current hyperparemeters and surrogate objective """
        self._log(sess)
        loss, _ = sess.run([self.loss, self.update_step])
        return loss
    
    def eval(self, sess):
        """ Evalute actual objective -- eg measure accuracy on the hold-out set """
        
        return sess.run(self.objective)
    
    def exploit(self, population ,sess):
        """ Copy theta from best member of the population """
        
        current_scores = [{
            "id": worker.id,
            "score": worker.eval(sess)
        } for worker in population]
        
        best_worker = sorted(current_scores, key=lambda x: x['score'])[-1]
        
        if best_worker['id'] != self.id:
            theta = sess.run(population[best_worker['id']].theta)
            sess.run(self.theta_assign, feed_dict={self.theta_:theta})
    
    def explore(self, sess, sd=0.1):
        """ Add normal noise to hyperparameter vector """
        
        h = sess.run(self.h) + np.random.random() * sd
        sess.run(self.h_assign, feed_dict={self.h_: h})


# In[ ]:


def run_experiment(do_explore=False, do_exploit=False, interval=5, n_steps=200):
    
    # Define objective functions
    objective = lambda theta: tf.constant(1.2) - tf.reduce_sum(tf.square(theta))
    surrogate_objective = lambda theta, h: tf.constant(1.2) - tf.reduce_sum(tf.square(h*theta))
    
    # Create population
    population = [
        Worker(
            theta=np.array([0.9,0.9]),
            h=np.array([1.0,0.0]),
            objective=objective,
            surrogate_objective=surrogate_objective,
            id=0,
        ),
        Worker(
            theta=np.array([0.9,0.9]),
            h=np.array([0.0,1.0]),
            objective=objective,
            surrogate_objective=surrogate_objective,
            id=1,
        ),
    ]
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    # Train
    for step in range(n_steps):
        for worker in population:
            if not (step + 1) % interval:
                
                if do_exploit:
                    worker.exploit(population,sess)
                    
                if do_explore:
                    worker.explore(sess)
            
            worker.step(sess)
    
    return population


# In[ ]:


# Run experiments w/ various PBT settings
pbt = run_experiment(do_explore=True, do_exploit=True) # Explore and exploit
explore = run_experiment(do_explore=True, do_exploit=False) # Explore only
exploit = run_experiment(do_explore=False, do_exploit=True) # Exploit only
grid = run_experiment(do_explore=False, do_exploit=False) # Independent training runs -- eg, regular grid search


# In[ ]:


def plot_score(ax, workers, run_name):
    """ Plot performance """
    for worker in workers:
        history = worker.history
        _ = ax.plot(history['score'], label="%s worker %d" % (run_name, worker.id), alpha=0.5)
    
    _ = ax.set_title(run_name)
    _ = ax.set_ylim(-1, 1.3)
    _ = ax.axhline(1.2, c='lightgrey')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plot_score(ax1, pbt, 'pbt')
plot_score(ax2, explore, 'explore')
plot_score(ax3, exploit, 'exploit')
plot_score(ax4, grid, 'grid')
_ = plt.tight_layout(pad=1)
plt.show()


# In[ ]:


def plot_theta(ax, workers, run_name):
    """ Plot values of theta """
    for worker in workers:
        history = worker.history
        _ = ax.scatter(history['theta'][:,0], history['theta'][:,1], 
            s=2, alpha=0.5, label="%s worker %d" % (run_name, worker.id))
    
    _ = ax.set_title(run_name)
    _ = ax.set_xlim(0, 1)
    _ = ax.set_ylim(0, 1)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plot_theta(ax1, pbt, 'pbt')
plot_theta(ax2, explore, 'explore')
plot_theta(ax3, exploit, 'exploit')
plot_theta(ax4, grid, 'grid')
_ = plt.tight_layout(pad=1)
plt.show()


# In[ ]:


def plot_h(ax, workers, run_name):
    """ Plot values of h"""
    for worker in workers:
        history = worker.history['h']
        _ = ax.scatter(history[:,0], history[:,1], s=2, alpha=0.5, label="%s worker %d"%(run_name, worker.id))
    
    _ = ax.set_title(run_name)
    _ = ax.set_xlim(0,3)
    _ = ax.set_ylim(0,3)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plot_h(ax1, pbt, 'pbt')
plot_h(ax2, explore, 'explore')
plot_h(ax3, exploit, 'exploit')
plot_h(ax4, grid, 'grid')
_ = plt.tight_layout(pad=1)
plt.show()

