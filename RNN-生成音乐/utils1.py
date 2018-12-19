"""
MIDI相关函数
"""
import os
import subprocess
from music21 import converter,instrument,note,chord,stream#音符和和弦
import pickle#用来读文件的操作
import glob#
def convertMidi2Mp3():

    """
    将MIDI转换为MP3
    :return:
    """
    input_file='./abc.mid'
    output_file="output.mp3"
    assert os.path.exists(input_file)

   # print("转换到MP3"%input_file)

    #用timidity生成MP3文件
    command='timidity{}  -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(input_file,output_file)
    subprocess.call(command,shell=True)


    print("转换文件为")

def get_notes():

    #从存放mide的文件夹读取数据存到data文件夹
    #读取NIDI文件，输出Stream流类型
    #读取所有midi文件读取note和chord
    #Note样例：A,B,A#,B#
    #CHor样例：【B4,E5,G#5],[C5 E5]
    #因为Chord就是多个Note的集合，所以我们简单的把他们统称为Note

    notes=[]
    #读取midi文件
    #glob：匹配所有符合条件的文件，并以List形式返回
    for file in glob.glob("./MIDI/*.mid"):
        strem=converter.parse(file)
        # 获取所有的乐器部分
        parts = instrument.partitionByInstrument(strem)

        if parts:
           notes_to_parse=parts.parts[0].recurse()#取第一个乐器部分
        else:
           notes_to_parse=strem.flat.notes

        for elment in notes_to_parse:
            #如果是NOTE类型，那么取它的音调pitch
            if isinstance(elment,note.Note):
                #格式   E6
                notes.append(str(elment.pitch))
            elif isinstance(elment,chord.Chord):
                #音调比较多就不用str处理了
        #将和弦转为整数，
                #转换后  4.1.2.3.34.   对于每一个音调有唯一一个整数值与之对应
                notes.append('.'.join(str(n) for n in elment.normalOrder))
        #将数据写入data文件
    with open('./Data/data1/1.txt','wb') as  filepath:
        pickle.dump(notes,filepath)


    return notes

def create_music(prediction):
    #用神经网络预测的音乐数据来生成MIDI文件，再转成mp3文件
   offset=0#表示偏移
   output_notes=[]

#生成NOTE音符或者Chord和弦对象
   for data in prediction:
        #是Chord格式例如4，15，7
      if('.'in data)or data.isdigit():#一个或者多个
           notes_in_chord=data.split('.')
           notes=[]
           for current_note in notes_in_chord:
               new_note=note.Note(int(current_note))
               new_note.storedInstrument=instrument.Piano()#乐器用钢琴
               notes.append(new_note)
           new_chord=chord.Chord(notes)
           new_chord.offset=offset
           output_notes.append(new_chord)
      else:
           new_note=note.Note(data)
           new_note.offset=offset
           new_note.storedInstrument=instrument.Piano()
           output_notes.append(new_note)

      #每次迭代都将偏移增加，这样才不会交叠覆盖
      offset+=0.5
   #创建音乐流
   midi_stream=stream.Stream(output_notes)
     #写入MIDI
   midi_stream.write('midi',fp='./output.mid')















if __name__=="__main__":
    convertMidi2Mp3()