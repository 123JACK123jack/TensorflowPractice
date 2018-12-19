from method  import *
from RNNmodel import *


def  test(model_path,test_data,vocab_size,id_to_word):
    test_input=Input(batch_size=20,num_steps=35,data=test_data)
    #基本的超参数要和训练时一致
    m=Model(test_input,is_training=False,hidden_size=650,vocab_size=vocab_size,num_layers=2)
    saver=tf.train.Saver()
    with tf.Session()  as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        current_state=np.zeros((2,2,m.batch_size,m.hidden_size))
        saver.restore(sess,model_path)

        num_acc_batches=30

        check_batch_idx=25

        acc_check_thresh=5
        accuracy=0
        for batch in range(num_acc_batches):
            if batch==check_batch_idx:
                true,pred,current_state,acc=sess.run([m.input_obj.targets,m.predict,m.state,m.accuracy],
                                                     feed_dict={m.init_state:current_state})
                pred_words=[id_to_word[x] for x in pred[:,m.num_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print(" ".join(true_words))
                print(" ".join(pred_words))
                coord.request_stop()
                coord.join(threads)


if __name__=="__main__":
    if args.data_path:
        data_path=args.data_path
    if args.load_file:
        load_file=args.load_file
        train_data,valid_data,test_data,vocab_size,id_to_word=load_data(data_path)
        trained_model=save_path
        test(trained_model,test_data,vocab_size,id_to_word)










