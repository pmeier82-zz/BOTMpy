#############
Spike Sorting
#############

Spike sorting is a combined detection and classification task for transient
patterns in time series. In neuroscience this task is found when analysing
voltage traces recorded from the brain of mammals. This kind of recording
yields discrete time series of a differential voltage trace, recorded from the
location in the cortex under consideration and a grounded electrode. Single
cells in the cortex (neurons) communicate by means of electrical and chemical
signals. These signals are called action potential (AP), the time course of the
membrane potential during an AP, as measured in the voltage trace recorded, has
a very distinct gestalt that gives rise to the term *spike*.

The aim of the spike sorting task is to identify the points in time when an
AP has occurred in the signal (spike detection) and to predict which of the
neurons has emitted this spike (spike sorting).

Extracellular recording
=======================

The leading paradigm to record an ensemble of neurons from the cortex is that
of extracellular recording. An electrode will be

.. _`fig-recording`:

.. figure:: static/recording.png
   :alt: cartoon of the experimental setup for extracellular recordings
   :align: center
   :figwidth: 80%
   :height: 300px

   Cartoon of the experimental setup for extracellular recordings.

Voltage patterns correspnding to action potentials of single cells have to be
found, and labeled as originating from that cell in a reliable manner. The
product will be a spike train that only carries the time and label for each
event, instead of the whole voltage transient from the recording.

For further information about spike sorting please refer to the following
publications *[Lewiki99]*, *[Scholarpedia]* and more recent and recommended
*[Einevoll11]*.

.. _template-matching:

Template Matching
-----------------
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi bibendum, neque
eu hendrerit scelerisque, orci nisl auctor risus, pulvinar congue augue turpis
fermentum odio. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam
venenatis lacinia elit, id aliquet dolor ultricies non. Sed quam massa,
ullamcorper sit amet scelerisque et, volutpat nec erat. Curabitur tincidunt
scelerisque dolor sit amet bibendum. Class aptent taciti sociosqu ad litora
torquent per conubia nostra, per inceptos himenaeos. Cras fermentum hendrerit
mattis. Nam ullamcorper nisl lacinia tortor suscipit sed iaculis augue
dignissim. Integer magna leo, pulvinar a pellentesque in, tincidunt quis lacus.
Donec et urna iaculis elit mollis venenatis. Maecenas a enim vitae arcu semper
ultrices condimentum eu justo. In hac habitasse platea dictumst. Maecenas in
felis quis enim malesuada laoreet.

Bayes Optimal Template Matching
-------------------------------
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi bibendum, neque
eu hendrerit scelerisque, orci nisl auctor risus, pulvinar congue augue turpis
fermentum odio. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam
venenatis lacinia elit, id aliquet dolor ultricies non. Sed quam massa,
ullamcorper sit amet scelerisque et, volutpat nec erat. Curabitur tincidunt
scelerisque dolor sit amet bibendum. Class aptent taciti sociosqu ad litora
torquent per conubia nostra, per inceptos himenaeos. Cras fermentum hendrerit
mattis. Nam ullamcorper nisl lacinia tortor suscipit sed iaculis augue
dignissim. Integer magna leo, pulvinar a pellentesque in, tincidunt quis lacus.
Donec et urna iaculis elit mollis venenatis. Maecenas a enim vitae arcu semper
ultrices condimentum eu justo. In hac habitasse platea dictumst. Maecenas in
felis quis enim malesuada laoreet.

References:
-----------

.. _`Python`: http://python.org/
.. _`Bayes Optimal Template Matching`:
  http://opus.kobv.de/tuberlin/volltexte/2012/3387/
.. [Lewiki99] "A review of methods for spike sorting: the detection and
              classification of neural action potentials"
              M. S. Lewicki (1998), Network: Computation in Neural Systems,
              Vol. 9, No. 4. (1998), pp. 53-78
.. [Scholarpedia] Rodrigo Quian Quiroga (2007) Spike sorting.
                  Scholarpedia, 2(12):3583.
.. [Einevoll11] "Towards reliable spike-train recordings from thousands of
                neurons with multielectrodes."
                Einevoll GT, Franke F, Hagen E, Pouzat C, Harris KD (2011),
                Curr Opin Neurobiol. 2011 Oct 22.
