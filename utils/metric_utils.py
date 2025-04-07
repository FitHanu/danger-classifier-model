import tensorflow as tf
from tensorflow.keras import backend as K

# Custom F1-score function
def f1_score(y_true, y_pred):
    """
    expected input shape:
    - y_true: (batch_size, num_classes)
    - y_pred: (batch_size, num_classes)
    """
    
    
    # This one convert multi dimen into one class indices array
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)  

    # Compute Precision and Recall manually
    # tp = K.sum(K.cast(y_true * y_pred, 'float32'))
    # fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    # fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))
    
    # Compute True Positives, False Positives, and False Negatives
    tp = K.sum(K.cast(y_true == y_pred, 'float32'))  # True Positives
    fp = K.sum(K.cast((y_true != y_pred), 'float32'))  # False Positives
    fn = K.sum(K.cast((y_true != y_pred), 'float32'))  # False Negatives

    precision = tp / (tp + fp + K.epsilon())  # Avoid division by zero
    recall = tp / (tp + fn + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())



if __name__ == "__main__":
    # Test metric function
    y_true = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=tf.float32)
    
    y_pred = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=tf.float32)
    
    # y_true = 
    # y_pred = tf.argmax(y_pred, axis=-1)
    print(tf.argmax(y_true, axis=-1))
    print(tf.argmax(y_pred, axis=-1))
    
    print(f"f1 score: {f1_score(y_true, y_pred)}")