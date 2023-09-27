"""
Pipeline.

--------


A Set of algorithms for cleaning the textual data into
something that is easier for the computer to read and
analyze.
"""

import json
import string
import re
from random import randint
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout, GRU
from tensorflow.keras.models import Model

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.networks import sequential
from tf_agents.drivers import py_driver, dynamic_step_driver
from tf_agents.policies import py_tf_eager_policy, random_tf_policy, PolicySaver
from tf_agents.metrics import tf_metrics
import reverb

PAD = '| '
version=5

def _main():
    word_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]
    #utils.validate_py_environment(environment, episodes=20) # Does not pass because the specs are not being followed through
    with open("data/words_250000_train.txt", 'r') as fp:
        vocab = fp.readlines()
        fp.close()
    vocab = [word.replace("\n", "") for word in vocab]
    environment = HangmanEnvironment(vocab)
    eval_environment = HangmanEnvironment(word_list)
    utils.validate_py_environment(environment, episodes=10) # Does not pass because the specs are not being followed through
    word_lengths = [len(word) for word in vocab]
    # Hyperparameters
    num_iterations = len(vocab)
    initial_collect_steps = 10
    collect_steps_per_iteration = 10
    replay_buffer_max_length = 100_000

    batch_size = 64
    learning_rate = 1e-3
    log_interval = 10

    num_eval_episodes = 10
    eval_interval = 200
    train_env = tf_py_environment.TFPyEnvironment(environment)
    eval_env = tf_py_environment.TFPyEnvironment(eval_environment)
    model = sequential.Sequential([
        Dense(28, activation="tanh", input_shape=()),
        Dense(52, activation="tanh"),
        Dense(104, activation="tanh"),
        Dense(78, activation="tanh"),
        Dense(52, activation="tanh"),
        Dense(26, activation="softmax")
        ], input_spec=tf.TensorSpec(shape=(28,)))

    train_step_counter = tf.Variable(0)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network = model,
            optimizer=opt,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
            )
    agent.initialize()
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    ## Get Replay Buffer
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()
            ]
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      replay_buffer.py_client,
      table_name,
      sequence_length=2)

    ## Train the agent

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.

    # Reset the environment.
    time_step = environment.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
          agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)
    #collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, collect_policy, observers=[replay_buffer.add_batch] + train_metrics, num_steps=collect_steps_per_iteration)

    # Dataset generates trajectories
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    losses = list()

    start = time.time()
    for _ in range(1_000):

         # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

         # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        losses.append(str(int(train_loss)))

        step = agent.train_step_counter.numpy()

        #if step % log_interval == 0:
        #    print('episode = {0}: loss = {1}'.format(step/10, train_loss))
        if time_step.is_last():
            print('iteration = {0}: loss = {1}'.format(step, train_loss))

        #if step % 1000 == 0:
        #    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        #    print('step = {0}: Average Return = {1}'.format(step, avg_return))
        #    returns.append(avg_return)
    # Setup Policy Saver
    with open("data/policy_{}.txt".format(version), 'w') as fp:
        fp.writelines(losses)
        fp.close()
    end = time.time()
    time_taken = end - start
    print("Time taken this loop (in seconds): {}".format(time_taken))
    print("Time taken this loop (in minutes): {}".format(time_taken/60))
    saver = PolicySaver(agent.collect_policy, batch_size=None)
    saver.save("models/policy_{}".format(version))


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def clean_text(text:str, pad:str = '|') -> str:
    """
    Clean Text to ASCII standard.

    -----------------------------

    Remove any special characters that do not add or
    provide context within the text. Also organizes
    sentences such that they do not possess any extra
    spaces and allows for proper context in terms of
    punctuation.

    Parameters
    ----------
    text : String
        Text in its rawest form, created by simply loading
        the text from the .txt file. This text should be
        loaded in using UTF-8 encoding.

    pad : String
        Padding to separate sample stories within the text.
    """
    text = text.lower()
    text = pad + text
    text = re.sub(r'\n{4,}', pad, text) # Need to change this
    text = text.replace('\n\n', '. ')
    text = text.replace('\n', ' ')
    text = text.replace(".'.", ".'")
    text = text.replace('!".', '"')
    text = text.strip()
    text = text.replace("..", ".")
    text = re.sub(r'([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace(' . ', '. ')
    return text

def create_vocabulary(text:str, delimiter:str=" ", pad:str="<pad>"):
    """
    Create a dictionary containing all of the words within the raw raw text indexed by numbers.

    ...

    Separates the words found within the clean body of text
    based on a predesignated delimiter (such as the space
    between the words) and assigns a number to the word.

    Parameters:
    -----------
    text : String
        Text after light processing. This is a singular
        string which has been modified to exclude special
        characters or anything that might not be consistent
        with english writing convention.

    delimiter : String
        Key used to separate the text into its separate
        entities or tokens.
    """
    tokens = text.split(delimiter)
    tokens.append(' ')
    unique_tokens = sorted(set(tokens))
    vocabulary = {i:unique_tokens[i] for i in range(len(unique_tokens))}
    return tokens, vocabulary

def generate_sequences(tokens:list, sequence_length:int, step:int):
    """
    Generate the dataset based on sequences of words.

    ...

    Uses a list of tokens to create the dataset by indexing
    the tokens based on sequence length.

    Parameters
    ----------
    tokens : List
        List containing words in the order of the story.

    sequence_length: Integer
        The distance into the future the model will predict.

    step : Integer
        The number of steps taken.
    """
    X = []
    y = []

    for i in range(0, len(set(tokens)) + 1 - sequence_length, step):
        X.append(tokens[i:i + sequence_length])
        y.append(tokens[i + sequence_length])

    y = to_categorical(y, num_classes = len(set(tokens)) + 1)
    num_seq = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, num_seq

def sample_with_temp(preds, temperature:float=1.0):
    """Sample an index from a probability array."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def generate_text(seed_text:str, next_words:int, model, tokenizer:Tokenizer, msl:int=20, temp:float=1.0) -> str:
    """
    Generate a body of text.

    ...

    Uses a generative machine learning model to create a
    story based on the a predetermined set of initial words
    and the total number of words desired. The determined
    number of words does not mean that the story will be as
    long, but rather that the max possible number of words
    will be equivalent to the variable `next words`. The
    msl allows the model to read the data in steps of n.

    Parameters
    ----------
    seed_text : String
        The beginning of the story. Predetermined by the
        user of the machine learning model.

    next_words : Integer
        The maximum possible number of words that the
        machine learning model will generate.

    model : TensorFlow Model
        Machine learning model used to generate text. Input
        text and the model will predict the next possible
        words.

    tokenizer : TensorFlow Tokenizer
        Model which converts tokens (words) to sequences.
        Part of TensorFlow's preprocessing algorithms.

    msl : Integer
        msl (Max Sequence Length) limits the length of the
        input to the machine learning model.

    temp : Floating Point Number
        Measure for the helper function to sample an index
        from a probability array.

    Returns
    -------
    output_text : String
        The complete story generated by the model.
    """
    output_text = seed_text
    start_story = PAD * msl
    seed_text = start_story + seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-msl:]
        token_list = np.reshape(token_list, (1, msl))

        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample_with_temp(probs, temp)

        output_word = tokenizer.index_word[y_class] if y_class > 0 else ''

        if output_word == "|":
            break

        seed_text += output_word + ' '
        output_text += output_word + ' '

    return output_text

def expand_dataset(filename:str) -> list[list]:
    """Expand the vocabulary based on guesses from hangman."""
    with open(filename, 'r') as fp:
        vocab = fp.readlines()
        fp.close()
    dataset = list()
    for word in vocab:
        word = word.replace("\n", "")
        iword = word
        combo = list()
        for letter in string.ascii_lowercase:
            combo.append(word)
            word = word.replace(letter, "_")
            if word == "_"*len(iword):
                continue
        dataset.append(combo)
    return dataset

class HangmanEnvironmentv1(py_environment.PyEnvironment):
    """This is the environment within the hangman game will be played."""

    def __init__(self, vocab:list):
        """Initialize the environment."""
        self.vocab = vocab
        self.word = np.random.choice(self.vocab)
        self.observation = [0] * (len(self.word)+1)
        self._episode_ended = False
        abcs = dict()
        for i, letter in enumerate(string.ascii_lowercase):
            abcs[letter] = i
        self.abc = abcs
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=25, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(len(self.word) + 1,), dtype=np.int32, minimum=0, maximum=26,name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        self._observation_spec = array_spec.BoundedArraySpec(shape=(len(self.word) + 1,), dtype=np.int32, minimum=0, maximum=26,name='observation')
        return self._observation_spec

    def _reset(self):
        #self.word = np.random.choice(self.vocab)
        self._episode_ended = False
        return ts.restart(np.array(self.observation, dtype=np.int32))

    def _get_observation(self):
        observation = np.zeros(len(self.word) + 1, dtype=np.int32)


    def _step(self, action):
        if self._episode_ended:
            print(" ")
            print(self.word)
            print(" ")
            return self.reset()

        if self.observation[-1] == 10:
            self._episode_ended = True
            return ts.termination(np.array(self.observation, dtype=np.int32), reward=0.0)
        else:
            pass
        sword = [self.abc[letter] for letter in self.word]
        if action in sword:
            indexes = {i:l for i, l in enumerate(self.word) if l == self.abc[l]}
            for i in indexes.keys():
                self.observation[i] = indexes[i]
            reward = len(indexes) / len(self.word)
            print("action is in sword.")
            return ts.transition(np.array(self.observation, dtype=np.int32), reward)
        elif self.observation[:-1] == sword:
            self._episode_ended = True
            final_reward = 1.0 - self.observation[-1]/10
            print("Action has ended.")
            return ts.termination(np.array(self.observation, dtype=np.int32),reward=final_reward)
        else:
            self.observation[-1] += 1
            discount = 1 / len(self.word)
            print("Failed {} times.".format(self.observation[-1]))
            return ts.transition(np.array(self.observation, dtype=np.int32), reward=0.0, discount = discount)

class HangmanEnvironment(py_environment.PyEnvironment):
    def __init__(self, word_list:list):
        self._word_list = word_list
        self._word = np.random.choice(self._word_list)
        self._guessed_letters = set()
        self._state = np.zeros(28, dtype=np.float32)
        self._state[-1] = len(self._word)
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=25, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(28,), dtype=np.float32, minimum=0, maximum=20, name='observation')
        self._discount_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=10.0, name='discount')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._word = np.random.choice(self._word_list, replace=False)
        self._state[-2] = int()
        self._state = np.zeros(28, dtype=np.float32)
        self._state[-1] = len(self._word)
        self._guessed_letters = set()
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        if self._state[-2] == 10: # Lose Case
            self._episode_ended = True
            discount = (sum(self._state[:-2]) - len(self._word)) / len(self._word)
            return ts.termination(self._state, reward=discount)
        if (set(self._guessed_letters) == set(self._word)) & (sum(self._state[:-1]) == len(self._word)): # Win Case
            self._episode_ended = True
            self._word_list.remove(self._word)
            reward = (sum(self._state[:-2]) - (sum(self._state[:-2]) * (self._state[-2]/ 10))) / len(self._word)
            return ts.termination(self._state, reward=reward)

        letter = chr(ord('a') + action)
        if letter in self._word: # Correct Guess Case
            pos = ord(letter) - 97
            self._state[pos] = self._word.count(letter)
            self._guessed_letters.add(letter)
            reward = (sum(self._state[:-2]) - self._state[-2]) / len(self._word)
            return ts.transition(self._state, reward=reward, discount=0.0)
        elif letter not in self._word: # Incorrect Guess Case
            self._state[-2] += 1
            discount = ((len(self._word) - sum(self._state[:-2])) * (self._state[-2] / 10)) / len(self._word)
            self._guessed_letters.add(letter)
            return ts.transition(self._state, reward=0.0, discount=abs(discount))


if __name__ == "__main__":
    _main()
#if __name__ == "__main__":
#    word_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]
#    environment = HangmanEnvironment(word_list)
#
#    # Reset the environment to start a new episode
#    time_step = environment.reset()
#    print("Initial Observation:", time_step.observation)
#
#    # Interact with the environment (e.g., take random actions)
#    for _ in range(10):
#        action = np.random.randint(26)  # Random letter (0-25)
#        time_step = environment.step(action)
#        print("Action:", chr(ord('a') + action))
#        print("Reward:", time_step.reward)
#        print("Observation:", time_step.observation)
#        if time_step.is_last():
#            break

