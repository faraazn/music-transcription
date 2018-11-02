import tensorflow as tf
import data

x_train, y_train = data.generate_samples(5000, 100)
x_test, y_test = data.generate_samples(1000, 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(84, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)