	???5??@???5??@!???5??@	???,??`????,??`?!???,??`?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???5??@ ??????A qW????@Yr??&OY??*?ʡE??I@)       =2U
Iterator::Model::ForeverRepeat?	MK??!h?%/@Q@)?8~?4b??1MŊt?E@:Preprocessing2_
(Iterator::Model::ForeverRepeat::Prefetchd*??g??!?F?s?:@)d*??g??1?F?s?:@:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2[|
????!?????8@)[|
????1?????8@:Preprocessing2F
Iterator::Model?:?????!F????R@)??#?Gk?1?D憹@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???,??`?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?????? ??????! ??????      ??!       "      ??!       *      ??!       2	 qW????@ qW????@! qW????@:      ??!       B      ??!       J	r??&OY??r??&OY??!r??&OY??R      ??!       Z	r??&OY??r??&OY??!r??&OY??JCPU_ONLYY???,??`?b 