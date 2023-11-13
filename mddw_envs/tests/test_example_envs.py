from mddw_envs.envs.example_envs import Env1a_2024
from mddw_envs.envs.example_envs import Env1b_2024
import unittest
   
class TestEnv1a_2024(unittest.TestCase):

    def test_init(self):
        env = Env1a_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        self.assertIsNotNone(env)

    def test_reset(self):
        env = Env1a_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        state, info = env.reset()
        self.assertIsNotNone(state)
        self.assertIsNotNone(info)

    def test_step(self):
        env = Env1a_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        state, info = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(done)
        self.assertIsNotNone(info)

class TestEnv1b_2024(unittest.TestCase):

    def test_init(self):
        env = Env1b_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        self.assertIsNotNone(env)

    def test_reset(self):
        env = Env1b_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        state, info = env.reset()
        self.assertIsNotNone(state)
        self.assertIsNotNone(info)

    def test_step(self):
        env = Env1b_2024(model_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339513/ibmracovid19modelv1.json", driver_data="https://github.com/Model-Driven-Discovery-Workshop/mddw-envs/files/13339558/location1.csv")
        state, info = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(done)
        self.assertIsNotNone(info)

