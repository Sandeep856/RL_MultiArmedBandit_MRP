import numpy as np
import matplotlib.pyplot as plt

class MarkovRewardProcess:
    def __init__(self):
      pass

    def rmse(self,V_true,V_estimate):
        return np.sqrt(np.mean((V_true - V_estimate) ** 2))

    def temporal_difference_learning(self,alpha,num_states,num_episodes):
        rmse_values = np.zeros(num_episodes)

        total_rewards=np.zeros(num_states)
        total_rewards[-1]=1
     

        V_true=np.zeros(num_states-2)
        for i in range(1,6):
          V_true[i-1]=i/6

        ActionCount=np.zeros((num_states,num_states))
        for episode in range(1000):# To smoothen the curve
            V = np.full(num_states-2, 0.5)

            for i in range(num_episodes):  # Limiting to 100 steps per episode
                state = num_states // 2

                while state!=0 and state!=num_states-1:
                    next_state = np.random.choice([state - 1, state + 1])
                    delta=0
                    if next_state==0 or next_state==num_states-1:
                      delta = total_rewards[next_state]- V[state-1]
                    else:
                      delta = total_rewards[next_state] + V[next_state-1] - V[state-1]

                    #ActionCount[state][next_state]+=1
                    V[state-1] +=alpha*delta
                    state = next_state

                rmse = self.rmse(V_true,V)
                rmse_values[i] += rmse / 100

        return rmse_values
    def func(self,num_states,num_episodes):
      rmse_values = np.zeros(num_episodes)

      total_rewards=np.zeros(num_states)
      total_rewards[-1]=1
      centerstate=num_states // 2

      V_true=np.zeros(num_states-2)
      for i in range(1,6):
        V_true[i-1]=i/6


      for episode in range(1000):# To smoothen the curve
          V = np.full(num_states, 0.5)
          V[0] = 0
          V[-1] = 0
          ActionCount=np.zeros((num_states,2))
          for i in range(num_episodes):  # Limiting to 100 steps per episode
              state = centerstate

              while state!=0 and state!=num_states-1:
                  next_state = np.random.choice([state - 1, state + 1])
                  action=0
                  if next_state==(state-1):
                    action=0
                  else:
                    action=1
                  delta = total_rewards[next_state] + V[next_state] - V[state]
                  ActionCount[state][action]+=1
                  V[state] +=delta/ActionCount[state][action]
                  state = next_state

              rmse = np.sqrt(np.mean((V_true - V[1:6]) ** 2))
              rmse_values[i] += rmse / 1000

      return rmse_values
    def plot_rmse_vs_episodes(self):
        plt.xlabel('Episodes')
        plt.ylabel('RMSE')
        plt.title('RMSE vs. Episodes')
        plt.grid()
        plt.show()







class Bandit:
    def __init__(self, no_of_arms, total_runs, num_steps):
        self.no_of_arms = no_of_arms
        self.total_runs = total_runs
        self.num_steps = num_steps

    def plotgraphs(self,xlabel,ylabel,title):
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.title(title)
      plt.legend()
      plt.grid
      plt.show()





    def run_UCB(self,c,alpha):

      ActualActionValues = np.zeros((self.total_runs,self.no_of_arms))
      ActionVal = np.zeros((self.total_runs,self.no_of_arms))
      Count_of_Actions = np.zeros((self.total_runs,self.no_of_arms))

      # True action values for the arms
      for i in range(self.total_runs):
          ActualActionValues[i] = np.random.normal(0, 1, self.no_of_arms)

      # Initialization arrays to store the percentage of optimal actions
      OptimalActions = np.zeros(1000)
      AveragRewards = np.zeros(self.num_steps)
      step=0
      while step < self.num_steps:
          count = 0
          var = 0
          for i in range(self.total_runs):
              # Implement the UCB algorithm
              if step < self.no_of_arms:
                  action = step
              else:
                  UCBVal = ActionVal[i] +c* np.sqrt(np.log(step + 1) / ((Count_of_Actions[i] + 1e-6)))
                  action = np.argmax(UCBVal)


              reward = np.random.normal(ActualActionValues[i][action], 1)

              Count_of_Actions[i][action] += 1
              var1=1
              ActionVal[i][action] += (reward - ActionVal[i][action])/Count_of_Actions[i][action]
              var+=reward

              if action == np.argmax(ActualActionValues[i]):
                  count += 1




          # Calculate the percentage of optimal actions
          AveragRewards[step] = var/2000
          OptimalActions[step] = (count / 2000)*100
          step+=1
      return [OptimalActions,AveragRewards]




    def run_optimistic(self,alpha,epsilon,qinit):


      AverageRewards = np.zeros(1000)
      ActualActionValues = np.zeros((self.total_runs,self.no_of_arms))
      ActionVal = np.full((self.total_runs,self.no_of_arms),qinit)

      for i in range(self.total_runs):
        ActualActionValues[i] = np.random.normal(0, 1, self.no_of_arms)


      OptimalActions = np.zeros(1000)


      step=0
      while step < self.num_steps:
        count =0
        total_rewards=0
        for i in  range(0,self.total_runs):

          action = np.argmax(ActionVal[i])

          reward = np.random.normal(ActualActionValues[i][action], 1)
          total_rewards+=reward

          ActionVal[i][action] += alpha * (reward - ActionVal[i][action])



          if action == np.argmax(ActualActionValues[i]):
              count += 1



        # Calculate the percentage of optimal actions for this run and step
        OptimalActions[step] = (count/2000)*100
        AverageRewards[step]=total_rewards/2000
        step+=1

      return [OptimalActions,AverageRewards]




    def run_epsilon_greedy(self,epsilon):
      AverageRewards=np.zeros(self.num_steps)

      ActualActionValues = np.zeros((self.total_runs, self.no_of_arms))
      ActionVal = np.zeros((self.total_runs, self.no_of_arms))
      Count_of_Actions = np.zeros((self.total_runs, self.no_of_arms))
      OptimalActions = np.zeros(1000)
      


      for i in range(self.total_runs):
        ActualActionValues[i] = np.random.normal(0, 1,self.no_of_arms)

      step=0
      while step<self.num_steps:
          count =0
          total_rewards=0
          for i in  range(0,self.total_runs):

            if np.random.rand() >= epsilon:

                    # Exploit
                    action = np.argmax(ActionVal[i])

            else:
                     # Explore
                     action = np.random.choice(self.no_of_arms)

            reward = np.random.normal(ActualActionValues[i][action], 1)
            total_rewards+=reward


            if action == np.argmax(ActualActionValues[i]):
                count += 1


            Count_of_Actions[i][action] += 1
            #ActionVal updated.
            ActionVal[i][action] += (reward - ActionVal[i][action])/Count_of_Actions[i][action]


          AverageRewards[step]=total_rewards/2000
          OptimalActions[step] = (count / 2000)*100
          step+=1
      return [OptimalActions,AverageRewards]




def main():
    no_of_arms = 10
    total_runs = 2000
    num_steps = 1000
    alpha = 0.1

    b = Bandit(no_of_arms,total_runs,num_steps)#object of bandit class

    #Epsilon greedy action value estimation method, Graphs plotted of both Optimal Actions taken and Average Rewards vs Steps
    epsilons = [0.1, 0.01, 0]  # Different epsilon values to test
    OptimalActions=[]
    AverageRewards=[]

    for epsilon in epsilons:
      var=b.run_epsilon_greedy(epsilon)
      OptimalActions.append(var[0])
      AverageRewards.append(var[1])


    for i in range(3):
      plt.plot(range(0,len(OptimalActions[i])),OptimalActions[i],label=f"Epsilon={epsilons[i]}")


    b.plotgraphs("Steps","%Optimal Actions taken","Percentage of Optimal Actions vs Steps-Epsilon greedy method")

    for i in range(3):
      plt.plot(range(0,len(AverageRewards[i])),AverageRewards[i],label=f"Epsilon={epsilons[i]}")


    b.plotgraphs("Steps","Average rewards","Rewards vs Steps-Epsilon greedy method")





    #Optimistic action value estimation method, Graphs plotted of both Optimal Actions taken and Average Rewards vs Steps
    var1=b.run_optimistic(alpha,0,5.0)
    var2=b.run_optimistic(0.2,0,5.0)
    var3=b.run_optimistic(0.3,0,5.0)
    plt.plot(range(0,len(var1[0])),var1[0],label="alpha=0.1")
    plt.plot(range(0,len(var2[0])),var2[0],label="alpha=0.2")
    plt.plot(range(0,len(var3[0])),var3[0],label="alpha=0.3")
    b.plotgraphs("Steps","%Optimal of action taken","Different values of alpha")

    #comparison of optimistic and epsilon greedy
    plt.plot(range(0,len(var1[0])),var1[0],label="Q1=5 and Epsilon=0")
    plt.plot(range(0,len(OptimalActions[0])),OptimalActions[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","% Optimal actions","Optimal actions vs steps-Optimistic method with alpha=0.1")

    #Average rewards
    plt.plot(range(0,len(var1[1])),var1[1],label="Q1=5 and Epsilon=0")
    plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","Average rewards","Average rewards vs steps-Optimistic method with alpha=0.1")


    #Optimistic action value estimation method, Graphs plotted of both Optimal Actions taken and Average Rewards vs Steps
    var=b.run_optimistic(alpha,0,2.0)
    plt.plot(range(0,len(var[0])),var[0],label="Q1=2 and Epsilon=0")
    plt.plot(range(0,len(OptimalActions[0])),OptimalActions[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","% Optimal actions","Optimal actions vs steps-Optimistic method with alpha=0.1")

    plt.plot(range(0,len(var[1])),var[1],label="Q1=2 and Epsilon=0")
    plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","Average rewards","Average rewards vs steps-Optimistic method with alpha=0.1")

    #Optimistic action value estimation method, Graphs plotted of both Optimal Actions taken and Average Rewards vs Steps
    var=b.run_optimistic(0.2,0,5.0)
    plt.plot(range(0,len(var[0])),var[0],label="Q1=5 and Epsilon=0")
    plt.plot(range(0,len(OptimalActions[0])),OptimalActions[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","% Optimal actions","Optimal actions vs steps-Optimistic method with alpha=0.2")

    plt.plot(range(0,len(var[1])),var[1],label="Q1=5 and Epsilon=0")
    plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","Average rewards","Average rewards vs steps -Optimistic method with alpha=0.2")







    #Upper confidence bound action value estimation method, Graphs plotted of both Optimal Actions taken and Average Rewards vs Steps with c=2
    var=b.run_UCB(2,0.1)
    var1=b.run_UCB(4,0.1)
    plt.plot(range(0,len(var[0])),var[0],label="c=2")
    plt.plot(range(0,len(var1[0])),var1[0],label="c=4")
    #plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","%Optimal Actions taken","Different values of UCB parameters")

    plt.plot(range(0,len(var[1])),var[1],label="c=2")
    plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","%Average Rewards","Average Rewards vs Steps-UCB and epsilon greedy")

    plt.plot(range(0,len(var[0])),var[0],label=f"c=2")
    plt.plot(range(0,len(OptimalActions[0])),OptimalActions[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","%Optimal Actions taken","%Optimal actions taken vs Steps-UCB method")

    #Upper confidence bound action value estimation method with the value of c as 4
    var=b.run_UCB(4,0.1)
    plt.plot(range(0,len(var[1])),var[1],label=f"c=4")
    plt.plot(range(0,len(AverageRewards[0])),AverageRewards[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","Average rewards","Average rewards vs steps-UCB method")

    plt.plot(range(0,len(var[0])),var[0],label=f"c=4")
    plt.plot(range(0,len(OptimalActions[0])),OptimalActions[0],label="Q1=0 and Epsilon=0.1")
    b.plotgraphs("Steps","%Optimal Actions taken","%Optimal actions taken vs Steps-UCB method")

    var=b.run_epsilon_greedy(0.1)
    var1=b.run_optimistic(0.1,0.1,5.0)
    var2=b.run_UCB(2,0.1)

    plt.plot(range(0,len(var[0])),var[0],label="Epsilon greedy")
    plt.plot(range(0,len(var1[0])),var1[0],label="Optimistic initial value")
    plt.plot(range(0,len(var2[0])),var2[0],label="UCB method")
    b.plotgraphs("Steps","%Optimal Action taken","Different Action selection methods in k armed bandit problem")




    num_states = 7
    alphas= [0.05,0.1,0.15,0.2]
    num_episodes = 100


    td_learning = MarkovRewardProcess()
    for a in alphas:
      rmse_values = td_learning.temporal_difference_learning(a,num_states,num_episodes)
      plt.plot(range(0,len(rmse_values)),rmse_values,label="Q1=0 and Epsilon=0.1")

    td_learning.plot_rmse_vs_episodes()
   
    rmse_values = td_learning.func(num_states,num_episodes)
    plt.plot(range(0,len(rmse_values)),rmse_values)

    td_learning.plot_rmse_vs_episodes()

if __name__ == "__main__":
    main()