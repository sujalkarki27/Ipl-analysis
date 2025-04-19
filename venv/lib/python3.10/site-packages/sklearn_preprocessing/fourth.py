def print_prog():
    print(
        """
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense

 

def split_sequences(sequences, n_steps):
          X, y = list(), list()
          for i in range(len(sequences)):
                     
                      end_ix = i + n_steps
                      
                      if end_ix > len(sequences)-1:
                                  break
                      
                      seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
                      X.append(seq_x)
                      y.append(seq_y)
          return array(X), array(y)

 

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps = 3
X, y = split_sequences(dataset, n_steps)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

"""
    )
