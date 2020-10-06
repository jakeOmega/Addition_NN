# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:20:57 2020

@author: jakef
"""
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Dense, Softmax, Flatten, LeakyReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.losses import categorical_crossentropy 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json

class learning_model:
    def __init__(self, characters):
        self.model_library = {}
        vectorizer = TextVectorization(standardize=lambda x: x , output_mode='int', split=None)
        vectorizer.adapt(np.array(list(characters)))
        self.vectorizer = vectorizer
        self.word_count = len(self.vectorizer.get_vocabulary())
        self.score_cache = {}
        
    def generate_new_model(self, input_size, output_size, hidden_layer_size, hidden_layer_count, sub_models = []):           
        input_layers = []
        for i in range(input_size):
            input_layers += [Input(shape=(self.word_count,))]
        flatten_layer = Flatten()(concatenate(input_layers))
        prev_layer = flatten_layer
        for layer_number in range(hidden_layer_count):
            sub_model_layers = []
            main_layer = LeakyReLU()(Dense(hidden_layer_size)(prev_layer))
            sub_model_index = 0
            for model_name, sub_model_layer_number in sub_models:
                if layer_number == sub_model_layer_number:
                    model = self.model_library[model_name]
                    model_inputs = len(model.input)
                    model_input_layers = [prev_layer[:, sub_model_index + i * self.word_count:sub_model_index + (i+1) * self.word_count] for i in range(model_inputs)]
                    model_layer = concatenate(model(model_input_layers))
                    sub_model_index += model_inputs * self.word_count
                    model_output_layer = Flatten()(model_layer)
                    sub_model_layers += [model_output_layer]                
            if len(sub_model_layers) == 0:
                prev_layer = main_layer
            else:
                prev_layer = concatenate([main_layer] + sub_model_layers)
        final_layers = []
        for i in range(output_size):
            output_layer = Dense(self.word_count)(prev_layer)
            final_layer = Softmax()(output_layer)
            final_layers += [final_layer]
        model = Model(input_layers, final_layers)
        model.compile(optimizer=Adamax(), loss=[categorical_crossentropy]*output_size)
        return model
        
    def add_model_to_library(self, model_name_input, input_size, output_size, hidden_layer_size, hidden_layer_count, sub_models = []):
        if model_name_input in self.model_library.keys():
            raise ValueError("This model name has already been used. Please choose another model name.")
        model  = self.generate_new_model(input_size, output_size, hidden_layer_size, hidden_layer_count, sub_models);
        self.model_library[model_name_input] = model
        
    def default_model_specification(self, input_size, output_size):
        model_spec = {
                "hidden_layer_size" : 100,
                "hidden_layer_count" : 0,
                "sub_models" : [],
                "input_size": input_size,
                "output_size": output_size
                }
        return model_spec
    
    def mutate(self, model_specification):
        valid_result = False
        while not valid_result:
            new_hidden_layer_size = max(0, min(1000, int(np.random.uniform(0.5, 1.5) * model_specification["hidden_layer_size"])))
            new_hidden_layer_count = max(0, model_specification["hidden_layer_count"] + np.random.choice([-1, 0, 1]))
            sub_model_change = np.random.choice(["add", "remove", "unchange"])
            if sub_model_change == "add":
                new_sub_models = [sub_model for sub_model in model_specification["sub_models"] if sub_model[1] < new_hidden_layer_count]
                if new_hidden_layer_count > 0 and len(self.model_library.keys()) > 0:
                    new_sub_models += [(np.random.choice(list(self.model_library.keys())), np.random.randint(0, new_hidden_layer_count))]
            elif sub_model_change == "remove":
                if len(model_specification["sub_models"]) > 0:
                    removal_index = np.random.randint(0, len(model_specification["sub_models"]))
                    new_sub_models = model_specification["sub_models"][:removal_index] + model_specification["sub_models"][removal_index+1:]
                else:
                    new_sub_models = []
            else:
                new_sub_models = model_specification["sub_models"]
                    
            new_model_spec = {
                    "hidden_layer_size" : new_hidden_layer_size,
                    "hidden_layer_count" : new_hidden_layer_count,
                    "sub_models" : new_sub_models,
                    "input_size" : model_specification["input_size"],
                    "output_size" : model_specification["output_size"]
                    
                    }
            valid_result = self.check_model_validity(new_model_spec)
        return new_model_spec
    
    def check_model_validity(self, model_spec):
        sub_models = model_spec["sub_models"]
        input_count = []
        for layer in range(model_spec["hidden_layer_count"]):
            layer_input = 0
            for sub_model in sub_models:
                if sub_model[1] == layer:
                    model = self.model_library[sub_model[0]]
                    layer_input += len(model.input) * self.word_count
            input_count += [layer_input]
        if model_spec["hidden_layer_count"] == 0:
            return True
        elif input_count[0] > model_spec["input_size"] * self.word_count or max([0] + input_count[1:]) > model_spec["hidden_layer_size"]:
            return False
        else:
            return True
    
    def evaluate(self, model_spec, input_string_list, output_string_list):
        key = json.dumps(model_spec, sort_keys=True, default=str)
        if key in self.score_cache.keys():
            return self.score_cache[key]
        input_size = model_spec["input_size"]
        output_size = model_spec["output_size"]
        hidden_layer_size = model_spec["hidden_layer_size"]
        hidden_layer_count = model_spec["hidden_layer_count"]
        sub_models = model_spec["sub_models"]
        model = self.generate_new_model(input_size, output_size, hidden_layer_size, hidden_layer_count, sub_models)
        input_vector = self.strings_to_vector(input_string_list)
        output_vector = self.strings_to_vector(output_string_list)
        history = model.fit(input_vector, output_vector, batch_size = 1000, epochs=10, validation_split = 0.5, verbose=0, use_multiprocessing=True, workers=8).history
        score = -history["val_loss"][-1]
        self.score_cache[key] = score
        return score
        
    def model_hyper_search(self, input_string_list, output_string_list, generations, population):
        model_pool = []
        for model_number in range(population):
            model_pool += [self.mutate(self.default_model_specification(len(input_string_list[0]), len(output_string_list[0])))]
        for generation in range(generations):
            scores = [self.evaluate(model, input_string_list, output_string_list) for model in model_pool]
            print(generation, scores[np.argmax(scores)], model_pool[np.argmax(scores)])
            scores = np.array([score - np.median(scores) for score in scores])
            keep_indices = np.where(scores >= 0)
            model_pool = [model_pool[i] for i in keep_indices[0]]
            print(len(model_pool))
            while len(model_pool) < population:
                model_pool += [self.mutate(np.random.choice(model_pool))]
        final_scores = [self.evaluate(model, input_string_list, output_string_list) for model in model_pool]
        return model_pool[np.argmax(final_scores)]
        
    def strings_to_vector(self, strings):
        if type(strings[0]) == bytes:
            strings = list(map(lambda x: x.decode("utf-8"), strings))
        string_list = list(map(list, strings))
        vector = list(tf.one_hot(self.vectorizer(string_list), self.word_count).numpy().swapaxes(0,1))
        return vector
    
    def vector_to_string(self, output):
        current_string = ''
        index_list = [np.argmax(output[i]) for i in range(len(output))]
        for i in index_list:
            current_string += self.vectorizer.get_vocabulary()[i]
        return current_string
    
    def model_fit(self, model_name, input_string_list, output_string_list, epochs=1):
        input_vector = self.strings_to_vector(input_string_list)
        output_vector = self.strings_to_vector(output_string_list)
        history = self.model_library[model_name].fit(input_vector, output_vector, verbose=1, batch_size = 1000, epochs=epochs, validation_split = 0.1, use_multiprocessing=True, workers=8)
        return history

    def lock_model(self, model_name):
        for layer in self.model_library[model_name].layers:
            layer.trainable = False
    
    

learning_model = learning_model(' 1234567890+')

size = 1000000
learning_model.add_model_to_library("basic_addition", 2, 2, 20, 1)
first_number = np.random.randint(0, 10, size=size)
second_number = np.random.randint(0, 10, size=size)
input_strings = np.core.defchararray.add(first_number.astype('|S1'),second_number.astype('|S1')).tolist()
output_strings = np.core.defchararray.rjust((first_number + second_number).astype('|S2'), 2, ' ').tolist()
learning_model.model_fit("basic_addition", input_strings, output_strings, epochs=30)
learning_model.lock_model("basic_addition")
tf.keras.utils.plot_model(learning_model.model_library["basic_addition"])
for string in ["11", "55"]:
    print(string, learning_model.vector_to_string(learning_model.model_library["basic_addition"](learning_model.strings_to_vector([string[:2].rjust(2)]))))

size = 1000000
input_single_size = 30
input_size = 2*input_single_size+1
output_size = 2 * input_single_size
learning_model.add_model_to_library("extract_place", input_size, output_size, output_size * learning_model.word_count, 1)
for i in range(30):
    first_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
    second_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
    input_strings = np.core.defchararray.rjust(np.core.defchararray.add(np.core.defchararray.add(first_number, '+'), second_number), input_size).tolist()
    a = np.core.defchararray.rjust(first_number, input_single_size, '0').astype(str)
    b = np.core.defchararray.rjust(second_number, input_single_size, '0').astype(str)
    output_strings = [''.join(a[i][j]+b[i][j]  for j in range(input_single_size)) for i in range(size)]
    history = learning_model.model_fit("extract_place", input_strings, output_strings, epochs=1).history["loss"]
    print("iteration: ", i, "  loss: ", history[-1])
    for string in ["1+1", "50+50", "100+1", "1+100", "1000000+1000000"]:
        print(string, learning_model.vector_to_string(learning_model.model_library["extract_place"](learning_model.strings_to_vector([string[:input_size].rjust(input_size)]))))
learning_model.lock_model("extract_place")


#size = 100000
#input_size = 2*input_single_size+1
#output_size = int((input_size + 1)/2)
#first_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
#second_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
#input_strings = np.core.defchararray.rjust(np.core.defchararray.add(np.core.defchararray.add(first_number, '+'), second_number), input_size).tolist()
#output_strings = [str(int(first_number[i])+int(second_number[i])).rjust(output_size, ' ') for i in range(size)]

#best_model = learning_model.model_hyper_search(input_strings, output_strings, 100, 10)

    
size = 100000
input_size = 2*input_single_size+1
output_size = int((input_size + 1)/2)
first_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
second_number = [''.join(["{}".format(np.random.randint(0, 9)) for num in range(1, np.random.randint(1, input_single_size))]).lstrip('0') + "{}".format(np.random.randint(0, 9)) for i in range(size)]
input_strings = np.core.defchararray.rjust(np.core.defchararray.add(np.core.defchararray.add(first_number, '+'), second_number), input_size).tolist()
output_strings = [str(int(first_number[i])+int(second_number[i])).rjust(output_size, ' ') for i in range(size)]

input_size = 2*input_single_size+1
output_size = int((input_size + 1)/2)
learning_model.add_model_to_library("big addition A", input_size, output_size, input_size * learning_model.word_count, 2, input_single_size * [("basic_addition", 1)] + [("extract_place", 0)])
learning_model.add_model_to_library("big addition B", input_size, output_size, input_size * learning_model.word_count, 2)
learning_model.add_model_to_library("big addition C", input_size, output_size, 10 * input_size * learning_model.word_count, 2)
history_A = []
history_B = []
history_C = []
lin_reg = LinearRegression()

    
for i in range(100):
    print(i)
    for string in ["1+1", "50+50", "100+1", "1+100", "1000000+1000000"]:
        print(string, learning_model.vector_to_string(learning_model.model_library["big addition A"](learning_model.strings_to_vector([string[:input_size].rjust(input_size)]))))
        print(string, learning_model.vector_to_string(learning_model.model_library["big addition B"](learning_model.strings_to_vector([string[:input_size].rjust(input_size)]))))
        print(string, learning_model.vector_to_string(learning_model.model_library["big addition C"](learning_model.strings_to_vector([string[:input_size].rjust(input_size)]))))
    history_A += [learning_model.model_fit("big addition A", input_strings, output_strings, epochs=1).history]
    history_B += [learning_model.model_fit("big addition B", input_strings, output_strings, epochs=1).history]
    history_C += [learning_model.model_fit("big addition C", input_strings, output_strings, epochs=1).history]


history_list_A = [x["val_loss"] for x in history_A]
history_list_B = [x["val_loss"] for x in history_B]
history_list_C = [x["val_loss"] for x in history_C]
plt.plot([item for sublist in history_list_A for item in sublist])
plt.plot([item for sublist in history_list_B for item in sublist])
plt.plot([item for sublist in history_list_C for item in sublist])
plt.yscale("log")
plt.show()