?	QhY?7|?@QhY?7|?@!QhY?7|?@	l?!<T?v?l?!<T?v?!l?!<T?v?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$QhY?7|?@:>Z?1???A{?<dB{?@Y\ A?c???*	?G?zhc@2U
Iterator::Model::ForeverRepeat+??	h??!?C!?Z?T@)OGɫs??1?O?Y:?N@:Preprocessing2_
(Iterator::Model::ForeverRepeat::PrefetchTƿϸ??!yo??5@)Tƿϸ??1yo??5@:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2?{?5Z??!?????:)@)?{?5Z??1?????:)@:Preprocessing2F
Iterator::Model?????]??!KN??U@)4Lm???~?1 u?"U@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9l?!<T?v?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:>Z?1???:>Z?1???!:>Z?1???      ??!       "      ??!       *      ??!       2	{?<dB{?@{?<dB{?@!{?<dB{?@:      ??!       B      ??!       J	\ A?c???\ A?c???!\ A?c???R      ??!       Z	\ A?c???\ A?c???!\ A?c???JCPU_ONLYYl?!<T?v?b Y      Y@qp?R?`?X?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 