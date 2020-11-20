## 有bug，返回的数必须有变化，否则PSCAD会警告，导致输出始终为0
##Main code will only run one time

#
# num_states = 1
# num_actions = 1
#
# upper_bound = 5
# lower_bound = -5
#
#
# ## Define noise class for action output
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


## Define Buffer class
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter = self.buffer_counter + 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# @tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


## Define actor network function
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


## Define critic network function
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model


## Define policy function for taking action
def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


## ddpg function for returning to MATLAB-PSCAD
def ddpg(state_1, reward, Done, Simu_Step_In):
    import math
    import numpy as np
    import random
    import tensorflow as tf
    from tensorflow.keras import layers

    ## Environment setting
    num_states = 1
    num_actions = 1

    upper_bound = 5
    lower_bound = -5

    ## Training hyperparameters
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.0002
    actor_lr = 0.0001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.9
    # Used to update target networks
    tau = 0.0005

    buffer = Buffer(50000, 64)

    episodic_reward = 0
    episodic_reward_store = []
    ## Train or not Train
    Train = True

    ## Start Training
    if Train:
        if (Simu_Step_In == 0):
            ### Not the fist episode: Load the weights
            actor_model.load_weights("./PSCAD_actor/PSCAD_actor")
            critic_model.load_weights("./PSCAD_critic/PSCAD_critic")
            target_actor.load_weights("./PSCAD_target_actor/PSCAD_target_actor")
            target_critic.load_weights("./PSCAD_target_critic/PSCAD_target_critic")


            ### First episode: Save the weights
            actor_model.save_weights("./PSCAD_actor/PSCAD_actor")
            critic_model.save_weights("./PSCAD_critic/PSCAD_critic")
            target_actor.save_weights("./PSCAD_target_actor/PSCAD_target_actor")
            target_critic.save_weights("./PSCAD_target_critic/PSCAD_target_critic")
            # Save the experience buffer
            np.save('buffer_counter_store.npy', buffer.buffer_counter)
            np.save('state_buffer_store.npy', buffer.state_buffer)
            np.save('action_buffer_store.npy', buffer.action_buffer)
            np.save('reward_buffer_store.npy', buffer.reward_buffer)
            np.save('next_state_buffer_store.npy', buffer.next_state_buffer)
            # Save the episode reward for final plotting
            np.save('episodic_reward_store.npy', episodic_reward_store)




            ### First step in each episode: store the 'episodic_reward' and 'prev_state'
            np.save('episodic_reward.npy', episodic_reward)
            prev_state = np.array([state_1])
            np.save('prev_state.npy', prev_state)
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            np.save('tf_prev_state.npy', tf_prev_state)

            # Execute action and save it
            sampled_actions = tf.squeeze(actor_model(tf_prev_state))
            noise = ou_noise()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise

            # We make sure action is within bounds
            legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
            action = [np.squeeze(legal_action)]
            np.save('action.npy', action)
            # action = policy(tf_prev_state, ou_noise) ##卡在这一步，很奇怪，明明python运行没有问题(测试证明这是一个bug,我直接不用函数,直接写函数里面的code可以成功)
            # np.save('action.npy',action)

            action_1 = float(action[0])
            Simu_Step_Out = Simu_Step_In + 1
            # 把当前的动作执行到PSCAD

        else:

            # Load previous state and previous action
            prev_state = np.load('prev_state.npy')
            action = np.load('action.npy')
            state = np.array([state_1])

            # Load the weights
            actor_model.load_weights("./PSCAD_actor/PSCAD_actor")
            critic_model.load_weights("./PSCAD_critic/PSCAD_critic")

            target_actor.load_weights("./PSCAD_target_actor/PSCAD_target_actor")
            target_critic.load_weights("./PSCAD_target_critic/PSCAD_target_critic")

            buffer_counter_store = np.load('buffer_counter_store.npy')
            state_buffer_store = np.load('state_buffer_store.npy')
            action_buffer_store = np.load('action_buffer_store.npy')
            reward_buffer_store = np.load('reward_buffer_store.npy')
            next_state_buffer_store = np.load('next_state_buffer_store.npy')

            buffer.buffer_counter = int(buffer_counter_store)
            buffer.state_buffer = state_buffer_store
            buffer.action_buffer = action_buffer_store
            buffer.reward_buffer = reward_buffer_store
            buffer.next_state_buffer = next_state_buffer_store

            buffer.record((prev_state, action, reward, state))

            episodic_reward = np.load('episodic_reward.npy')
            episodic_reward = float(episodic_reward) + reward  ##此处也有一个bug,不能使用episodic_reward += Reward格式
            np.save('episodic_reward.npy', episodic_reward)

            # buffer.learn() ## 此处也有bug,所以把code从Buffer类中直接提取出来运行
            # Get sampling range
            record_range = min(buffer.buffer_counter, buffer.buffer_capacity)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, buffer.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(buffer.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(buffer.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(buffer.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(buffer.next_state_buffer[batch_indices])

            # buffer.update(state_batch, action_batch, reward_batch, next_state_batch) ## 此处也有bug,所以把code从Buffer类中直接提取出来运行

            # Training and updating Actor & Critic networks.
            # See Pseudo Code.
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_state_batch, training=True)
                y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
                critic_value = critic_model([state_batch, action_batch], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

            with tf.GradientTape() as tape:
                actions = actor_model(state_batch, training=True)
                critic_value = critic_model([state_batch, actions], training=True)
                # Used '-value' as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            np.save('buffer_counter_store.npy', buffer.buffer_counter)
            np.save('state_buffer_store.npy', buffer.state_buffer)
            np.save('action_buffer_store.npy', buffer.action_buffer)
            np.save('reward_buffer_store.npy', buffer.reward_buffer)
            np.save('next_state_buffer_store.npy', buffer.next_state_buffer)

            # Save the weights
            actor_model.save_weights("./PSCAD_actor/PSCAD_actor")
            critic_model.save_weights("./PSCAD_critic/PSCAD_critic")

            target_actor.save_weights("./PSCAD_target_actor/PSCAD_target_actor")
            target_critic.save_weights("./PSCAD_target_critic/PSCAD_target_critic")

            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            sampled_actions = tf.squeeze(actor_model(tf_state))
            noise = ou_noise()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise

            # We make sure action is within bounds
            legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
            action = [np.squeeze(legal_action)]
            np.save('action.npy', action)

            prev_state = state
            np.save('prev_state.npy', prev_state)

            if (Simu_Step_In == 5000):
                episodic_reward_store = np.load('episodic_reward_store.npy')
                episodic_reward_store = np.append(episodic_reward_store, episodic_reward)
                np.save('episodic_reward_store.npy', episodic_reward_store)

            action_1 = float(action[0])
            Simu_Step_Out = Simu_Step_In + 1

    ## Not train, just run
    else:

        # Load the weights
        actor_model.load_weights("./PSCAD_actor/PSCAD_actor")

        prev_state = np.array([state_1])
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        sampled_actions = tf.squeeze(actor_model(tf_prev_state))
        noise = ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # sampled_actions = sampled_actions.numpy()

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        action = [np.squeeze(legal_action)]
        # np.save('action.npy',action)
        # action = policy(tf_prev_state, ou_noise) ##卡在这一步，很奇怪，明明python运行没有问题(测试证明这是一个bug,我直接不用函数,直接写函数里面的code可以成功)
        # np.save('action.npy',action)

        action_1 = float(action[0])
        Simu_Step_Out = Simu_Step_In + 1

    return action_1, Simu_Step_Out

def add(a,b):
    import math
    import random
    import tensorflow as tf
    from tensorflow.keras import layers
    return a*b, a+b