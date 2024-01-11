import fileinput
import codecs, json
import sys
from pydub import AudioSegment
import numpy as np
import yaml
import os
from pyannote.audio.pipelines.segmentation import SpeakerSegmentation as Pipeline_seg
from pyannote.audio import Pipeline

path = os.path.dirname(os.path.realpath(__file__))

class SpeechProcessingUnit:
    def __init__(self, num_gpu, prob_threshold):
        # initialize segmentation pipeline
        self.args, self.path = self.initialize_config()
        self.pipeline_seg = Pipeline_seg(segmentation=f"{path}/models/segmentation/epoch=330-step=27804.ckpt",
                                         my_needs=num_gpu)
        self.initial_params = {"onset": float(prob_threshold)+0.1,
                               "offset": float(prob_threshold)-0.1,
                               "stitch_threshold": self.args['params']['stitch_threshold'],
                               "min_duration_on": self.args['params']['min_duration_on'],
                               "min_duration_off": self.args['params']['min_duration_off']
                               }
        self.pipeline_seg.instantiate(self.initial_params)

        # initialize diarization pipeline
        self.pipeline_dia = Pipeline.from_pretrained(f"{path}/models/pipeline/config.yml", num_gpu)

    def initialize_config(self):

        # path = "/app/bin/external_apps/ai_unit"

        self.path = path
        with open(f'{path}/config.yml') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        with open(f'{path}/models/pipeline/config.yml') as s:
            pipe = yaml.load(s, Loader=yaml.FullLoader)

        pipe['pipeline']['params']['embedding'] = f'{path}/models/pipeline/speechbrain'
        pipe['pipeline']['params']['segmentation'] = f'{path}/models/segmentation/epoch=330-step=27804.ckpt'

        with open(f'{path}/models/pipeline/config.yml', "w") as s:
            yaml.dump(pipe, s)

        for i, line in enumerate(fileinput.input(f'{path}/models/pipeline/speechbrain/hyperparams.yaml', inplace=1)):
            if line[0:15] == "pretrained_path":
                sys.stdout.write(line.replace(line, f'pretrained_path: "{path}/models/pipeline/speechbrain"\n'))
            else:
                sys.stdout.write(line.replace(line, line))

        return args, path

    def __call__(self, file: str, task: str) -> list:
        if task == "segmentation":
            segmentation, seg_annotation = self.segmentation_task(file)

            return seg_annotation
        elif task == "diarization":
            diarization = self.diarization_task(file)
            print(type(diarization))
            return diarization
        elif task == "segmentation & diarization":
            segmentation, seg_annotation = self.segmentation_task(file)
            diarization = self.diarization_task(file, segmentation)
            print(type(diarization))
            return [seg_annotation, diarization]
        else:
            sys.exit("your task should be one of segmentation, diarization, segmentation & diarization")


    def segmentation_task(self, file):
        output_seg = self.pipeline_seg(file, model_us=None)
        segments = output_seg[0]._tracks
        seg_annotation = []
        for seg in segments:
            seg_dict = {"begin": 0, "end": 0}
            seg_dict.update({"begin": round(seg.start, 2), "end": round(seg.end, 2)})
            seg_annotation.append(seg_dict)
        return output_seg, seg_annotation

    def diarization_task(self, file, segmentation_result=None):
        segmentation = None
        if segmentation_result is None:
            segmentation, seg_annotation = self.segmentation_task(file)
        else:
            segmentation = segmentation_result

        segmentation_for_dia = segmentation[2]
        output_dia = self.pipeline_dia(file, model_us=segmentation_for_dia)
        embeddings = output_dia[1]
        output_dia = output_dia[0]
        diarization = output_dia._tracks
        dia_annotation = []
        for dia in diarization.items():
            dia_dict = {"begin": 0, "end": 0, "speakers": " "}
            for i in dia[1].values():
                id_speakers = i
                dia_dict.update(
                    {"begin": round(dia[0].start, 2), "end": round(dia[0].end, 2), "speakers": f'{id_speakers}'})
                dia_annotation.append(dia_dict)

        output_annotation = {"dia_annotation": 0}
        output_annotation.update({"dia_annotation": dia_annotation})

        segments_probs = self.diarization_probabilities(diarization, embeddings, dia_annotation)

        for index, prob in enumerate(segments_probs):
            prob_list = []
            for lenght in range(len(prob)):
                prob_list.append({"id": lenght, "prob": round(prob[lenght] * 100, 2)})

            newlist = sorted(prob_list, key=lambda d: d['prob'], reverse=True)
            output_annotation['dia_annotation'][index]['speakers'] = newlist

        return [output_annotation, segments_probs]


    @staticmethod
    def diarization_probabilities(diarization: dict, embeddings: dict, dia_annotation: list) -> list:
        segments_embedding = []
        for dia in diarization:
            embed_sum = [0] * 192
            for embed in range(int(dia.start // 0.2), int(dia.end // 0.2) + 1):
                for code_embed in range(len(embeddings[embed][0])):
                    if not np.isnan(embeddings[embed][0][code_embed]):
                        embed_sum[code_embed] += embeddings[embed][0][code_embed]
                    if not np.isnan(embeddings[embed][1][code_embed]):
                        embed_sum[code_embed] += embeddings[embed][1][code_embed]
            embed_sum = np.array(embed_sum)
            embed_count = dia.end // 0.2 + 1 - dia.start // 0.2 + 1
            embed_end = embed_sum / int(embed_count)
            segments_embedding.append(embed_end)
        number_speakers = 0

        for dia_annot in range(len(dia_annotation)):
            speaker_number = int(dia_annotation[dia_annot]["speakers"][8:])
            if number_speakers < speaker_number:
                number_speakers = speaker_number
        number_speakers += 1
        cluster_centers = np.array([[0.0] * 192] * number_speakers)
        cluster_centers_assign = np.array([0] * number_speakers)

        for dia_annot in range(len(dia_annotation)):
            if float(dia_annotation[dia_annot]["end"] - dia_annotation[dia_annot]["begin"]) > 1:
                speaker_number = int(dia_annotation[dia_annot]["speakers"][8:])
                cluster_centers[speaker_number] += segments_embedding[dia_annot]
                cluster_centers_assign[speaker_number] += 1

        for n_speaker in range(number_speakers):
            cluster_centers[n_speaker] = cluster_centers[n_speaker] / cluster_centers_assign[n_speaker]
        segments_probs = []

        for seg_index in range(len(segments_embedding)):
            distance_from_center = []
            sum_all = 0
            for cluster_index in range(len(cluster_centers)):
                sum_sq = np.sqrt(np.sum(np.square(segments_embedding[seg_index] - cluster_centers[cluster_index])))
                distance_from_center.append(sum_sq)
                sum_all += sum_sq
            sum_temp = 0
            for distance in range(len(distance_from_center)):
                distance_from_center[distance] = sum_all / distance_from_center[distance]
                sum_temp += distance_from_center[distance]
            distance_from_center = np.array(distance_from_center)
            distance_from_center = distance_from_center / sum_temp
            segments_probs.append(distance_from_center)
        return segments_probs


if __name__ == "__main__":
    file_path_index = -1
    save_path_index = -1
    vad_path_index = -1
    threshold_index = -1
    gpu_number_index = -1
    task_index = -1

    threshold = 0.5
    save_path = ""
    file_path = ""
    vad_path = ""
    gpu_number = 0
    task = ""

    for i, arg in enumerate(sys.argv):
        if arg == "file_path":
            file_path_index = i + 1

        elif arg == "save_path":
            save_path_index = i + 1

        elif arg == "vad_path":
            vad_path_index = i + 1

        elif arg == "threshold":
            threshold_index = i + 1

        elif arg == 'gpu_number':
            gpu_number_index = i + 1

        elif arg == "task":
            task_index = i+1

        elif vad_path_index == i:
            vad_path = arg

        elif file_path_index == i:
            file_path = arg

        elif save_path_index == i:
            save_path = arg

        elif threshold_index == i:
            threshold = arg

        elif gpu_number_index == i:
            gpu_number = int(arg)

        elif task_index == i:
            task = arg


    speech = SpeechProcessingUnit(gpu_number, threshold)
    testdata_dir = file_path
    if (testdata_dir.split(".")[-1]) == "mp3":
        sound = AudioSegment.from_mp3(testdata_dir)
        sound.export(f"{speech.path}/sound.wav", format="wav")
        testdata_dir = f'{speech.path}/sound.wav'
        print("===start===")
        print(json.dumps(speech(testdata_dir, task)))
        print("====end====")
    else:
        print("===start===")
        print(json.dumps(speech(testdata_dir, task)))
        print("====end====")
