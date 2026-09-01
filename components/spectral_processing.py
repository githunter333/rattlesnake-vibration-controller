# -*- coding: utf-8 -*-
"""
Controller subsystem that handles computation of FRFs, CPSDs, and other spectral
quantities of interest

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import multiprocessing as mp
from .utilities import (flush_queue,DataAcquisitionParameters,VerboseMessageQueue,GlobalCommands,
                        load_python_module)
import scipy.signal as sig
import numpy as np
from enum import Enum
from .abstract_message_process import AbstractMessageProcess
import time
import os

# Set True to enable spectral_processing.py's CVA raw-data capture hook
# (dumps the raw sys-ID response/reference buffers + resulting FRF to
# examples/sixdrive12resp/results/cva_captures/latest_cva_sysid_capture.npz
# on every successful CVA fit, for offline verification -- see the comment
# at the capture site in _run_cva_processing). Off has zero cost/behavior
# change; on writes one small side file per fit.
CVA_CAPTURE_RAW_DATA = True

# Set True to enable spectral_processing.py's H1/H2/H3/HV raw-FRF capture
# hook (dumps the resulting FRF + frequencies + underlying cross/auto
# spectral matrices to examples/sixdrive12resp/results/cva_captures/
# latest_h1_sysid_capture.npz on every successful non-CVA FRF computation,
# for offline verification against ground truth -- see the comment at the
# capture site in run_spectral_processing, right before the FRF is put on
# data_out_queue). Off has zero cost/behavior change; on writes one small
# side file per fit. Named "h1" for brevity but fires for whichever
# estimator (H1/H2/H3/HV) is actually selected -- check the "estimator"
# field in the saved file.
H1_CAPTURE_FRF = True

WAIT_TIME = 0.05

class SpectralProcessingCommands(Enum):
    """Collection of instructions that the FRF Computation Process might get"""
    INITIALIZE_PARAMETERS = 0
    RUN_SPECTRAL_PROCESSING = 1
    CLEAR_SPECTRAL_PROCESSING = 2
    STOP_SPECTRAL_PROCESSING = 3
    SENT_SPECTRAL_DATA = 4
    SHUTDOWN_ACHIEVED = 5

class AveragingTypes(Enum):
    LINEAR = 0
    EXPONENTIAL = 1

class Estimator(Enum):
    H1 = 0
    H2 = 1
    H3 = 2
    HV = 3
    CVA_INNOVATIONS = 4

class SpectralProcessingMetadata():
    def __init__(self,
                 averaging_type, averages,
                 exponential_averaging_coefficient,
                 frf_estimator,
                 num_response_channels,
                 num_reference_channels,
                 frequency_spacing,
                 sample_rate,
                 num_frequency_lines,
                 compute_cpsd = True,
                 compute_frf = True,
                 compute_coherence = True,
                 compute_apsd = True,
                 cva_lags = 40,
                 cva_rank = 66,
                 cva_window_seconds = 2.0,
                 cva_refine_iters = 1,
                 cva_refit_interval_seconds = 1.0):
        # cva_* : only consulted when frf_estimator == Estimator.CVA_INNOVATIONS.
        # Defaults are the offline-validated settings for the 6-drive/8-response
        # sixdrive12resp bench system (globalcva/ sweep results); per the design
        # doc's "not yet validated" note, these do NOT automatically transfer to
        # a different channel count -- a production system needs its own sweep.
        # cva_refit_interval_seconds is a simple wall-clock throttle (this pass
        # only -- real gating/publish logic is a separate, not-yet-built piece)
        # so a live CVA fit doesn't re-run every ~50ms polling cycle.
        self.averaging_type = averaging_type
        self.averages = averages
        self.exponential_averaging_coefficient = exponential_averaging_coefficient
        self.frf_estimator = frf_estimator
        self.num_response_channels = num_response_channels
        self.num_reference_channels = num_reference_channels
        self.frequency_spacing = frequency_spacing
        self.sample_rate = sample_rate
        self.num_frequency_lines = num_frequency_lines
        self.compute_cpsd = compute_cpsd
        self.compute_frf = compute_frf
        self.compute_coherence = compute_coherence
        self.compute_apsd = compute_apsd
        self.cva_lags = cva_lags
        self.cva_rank = cva_rank
        self.cva_window_seconds = cva_window_seconds
        self.cva_refine_iters = cva_refine_iters
        self.cva_refit_interval_seconds = cva_refit_interval_seconds

    def __eq__(self,other):
        try:
            return np.all([np.all(self.__dict__[field] == other.__dict__[field]) for field in self.__dict__])
        except (AttributeError,KeyError):
            return False
        
    @property
    def requires_full_spectral_response(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.H2, Estimator.H3])
            or self.compute_cpsd):
            return True
        else:
            return False
        
    @property
    def requires_diagonal_spectral_response(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.HV])
            or self.compute_apsd or self.compute_coherence):
            return True
        else:
            return False
        
    @property
    def requires_full_spectral_reference(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.H1, Estimator.H3, Estimator.HV])
            or self.compute_cpsd or self.compute_coherence):
            return True
        else:
            return False
    
    @property
    def requires_diagonal_spectral_reference(self):
        if self.compute_apsd:
            return True
        else:
            return False
    
    @property
    def requires_spectral_reference_response(self):
        if self.compute_frf or self.compute_coherence:
            return True
        else:
            return False

class SpectralProcessingProcess(AbstractMessageProcess):
    """Class defining a subprocess that computes a FRF from a time history."""
    def __init__(self,process_name : str, 
                 command_queue : VerboseMessageQueue,
                 data_in_queue : mp.queues.Queue,
                 data_out_queue : mp.queues.Queue,
                 environment_command_queue : VerboseMessageQueue,
                 gui_update_queue : mp.queues.Queue,
                 log_file_queue : mp.queues.Queue,
                 environment_name : str,
                 raw_data_in_queue : mp.queues.Queue = None):
        """
        Constructor for the FRF Computation Process
        
        Sets up the ``command_map`` and initializes internal data

        Parameters
        ----------
        process_name : str
            Name for the process that will be used in the Log file.
        command_queue : VerboseMessageQueue :
            The queue containing instructions for the FRF process
        data_for_frf_queue : mp.queues.Queue :
            Queue containing input data for the FRF computation
        updated_frf_queue : mp.queues.Queue :
            Queue where frf process will put computed frfs
        gui_update_queue : mp.queues.Queue :
            Queue for gui updates
        log_file_queue : mp.queues.Queue :
            Queue for writing to the log file
        environment_name : str
            Name of the environment that controls this subprocess.

        """
        super().__init__(process_name,log_file_queue,command_queue,gui_update_queue)
        self.map_command(SpectralProcessingCommands.INITIALIZE_PARAMETERS,self.initialize_parameters)
        self.map_command(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,self.run_spectral_processing)
        self.map_command(SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,self.clear_spectral_processing)
        self.map_command(SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING,self.stop_spectral_processing)
        self.environment_name = environment_name
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.environment_command_queue = environment_command_queue
        self.response_spectral_matrix = None
        self.reference_spectral_matrix = None
        self.response_reference_spectral_matrix = None
        self.reference_diagonal_matrix = None
        self.response_diagonal_matrix = None
        self.response_fft = None
        self.reference_fft = None
        self.spectral_processing_parameters = None
        self.frames_computed = 0
        # CVA (Estimator.CVA_INNOVATIONS) state -- unused, zero cost, for any
        # other estimator. raw_data_in_queue carries (raw_response,raw_reference)
        # un-windowed time blocks from DataCollectorProcess's raw tap (see
        # data_collector.py); the rolling buffers below accumulate them into
        # the sliding window global_cva_innovations fits.
        self.raw_data_in_queue = raw_data_in_queue
        self._cva_response_buffer = None   # (num_response_channels, N) rolling raw samples
        self._cva_reference_buffer = None  # (num_reference_channels, N)
        self._cva_last_fit_wall_time = 0.0
        self._cva_module = None            # lazily-loaded globalcva/global_cva_frf.py
        # Last successfully-fit FRF/coherence/condition, reused as a fallback
        # on a failed fit (e.g. during the zero-drive noise-floor phase,
        # where CVA's Hankel/covariance matrices are exactly singular --
        # expected and harmless, see _run_cva_processing). None until the
        # first successful fit.
        self._cva_last_frf = None
        self._cva_last_coherence = None
        self._cva_last_condition = None
           
    def initialize_parameters(self,data : SpectralProcessingMetadata):
        """Initializes the signal processing parameters from the environment.

        Parameters
        ----------
        data :
            Container containing the setting specific to the environment.

        """
        if self.spectral_processing_parameters is None:
            reshape_arrays = True
        elif (self.spectral_processing_parameters.num_frequency_lines !=
              data.num_frequency_lines
              or
              self.spectral_processing_parameters.num_response_channels !=
              data.num_response_channels
              or
              self.spectral_processing_parameters.num_reference_channels !=
              data.num_reference_channels
              or
              self.spectral_processing_parameters.averages != 
              data.averages
              or self.spectral_processing_parameters.averaging_type !=
              data.averaging_type):
            reshape_arrays = True
        else:
            reshape_arrays = False
        self.spectral_processing_parameters = data
        if reshape_arrays:
            self.log('Initializing Empty Arrays')
            self.frames_computed = 0
            self._cva_response_buffer = None
            self._cva_reference_buffer = None
            self._cva_last_fit_wall_time = 0.0
            self._cva_last_frf = None
            self._cva_last_coherence = None
            self._cva_last_condition = None
            self.response_spectral_matrix = None
            self.reference_spectral_matrix = None
            self.reference_response_spectral_matrix = None
            self.reference_diagonal_matrix = None
            self.response_diagonal_matrix = None
            if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
                self.response_fft = np.nan*np.ones((
                    self.spectral_processing_parameters.averages,
                    self.spectral_processing_parameters.num_response_channels,
                    self.spectral_processing_parameters.num_frequency_lines),dtype=complex)
                self.reference_fft = np.nan*np.ones((
                    self.spectral_processing_parameters.averages,
                    self.spectral_processing_parameters.num_reference_channels,
                    self.spectral_processing_parameters.num_frequency_lines),dtype=complex)
                # print(self.response_fft.shape)
            else:
                self.response_fft = None
                self.reference_fft = None
    
    def run_spectral_processing(self,data):
        """Continuously compute FRFs from time histories.
        
        This function accepts data from the ``data_for_frf_queue`` and computes
        FRF matrices from the time data.  It uses a rolling buffer to append
        data.  The oldest data is pushed out of the buffer by the newest data.
        The test level is also passed with the response data and output
        data.  The test level is used to ensure that no frame uses
        discontinuous data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        if (self.spectral_processing_parameters is not None
                and self.spectral_processing_parameters.frf_estimator == Estimator.CVA_INNOVATIONS):
            return self._run_cva_processing(data)
        data = flush_queue(self.data_in_queue,timeout = WAIT_TIME)
        if len(data) == 0:
            time.sleep(WAIT_TIME)
            self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
            return
        frames_received = len(data)
        self.log('Received {:} Frames'.format(frames_received))
        if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
            response_fft,reference_fft = [value for value in zip(*data)]
            self.response_fft = np.concatenate((self.response_fft[frames_received:],response_fft[-self.response_fft.shape[0]:]),axis=0)
            self.reference_fft = np.concatenate((self.reference_fft[frames_received:],reference_fft[-self.reference_fft.shape[0]:]),axis=0)
            self.log('Buffered Frames (Resp Shape: {:}, Ref Shape: {:})'.format(self.response_fft.shape,self.reference_fft.shape))
            # Exclude any with NaNs
            exclude_averages = np.any(np.isnan(self.response_fft),axis=(-1,-2))
            self.log('Computed Number Averages {:}'.format((~exclude_averages).sum()))
            # Return if there is actually no data
            if np.all(exclude_averages):
                self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
                return
            self.log('Mean FFT Value Over Averaged Frames: \n  {:}'.format(np.mean(np.abs(self.reference_fft[~exclude_averages]),axis=(-1,-2))))
            # Now we compute the spectral matrices depending on what is required.
            # Compute the response power spectra
            response_spectral_time = time.time()
            if self.spectral_processing_parameters.requires_full_spectral_response:
                self.log('Computing Full Spectral Response Matrix')
                self.response_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.response_fft[~exclude_averages],
                    np.conj(self.response_fft[~exclude_averages])
                    )/self.response_fft[~exclude_averages].shape[0]
                # Get the diagonal matrix as well
                self.response_diagonal_matrix = np.einsum(
                    'fii->fi',
                    self.response_spectral_matrix)
            elif self.spectral_processing_parameters.requires_diagonal_spectral_response:
                self.log('Computing Diagonal of Spectral Response Matrix')
                # self.response_diagonal_matrix = np.einsum(
                #     'aif,aif->fi',
                #     self.response_fft[~exclude_averages],
                #     np.conj(self.response_fft[~exclude_averages])
                #     )/self.response_fft[~exclude_averages].shape[0]
                self.response_diagonal_matrix = np.mean(self.response_fft[~exclude_averages]*np.conj(self.response_fft[~exclude_averages]),axis=0).T
            if (self.spectral_processing_parameters.requires_full_spectral_response
                or self.spectral_processing_parameters.requires_diagonal_spectral_response):
                self.log('Computed Response Spectral Matrix in {:0.2f} seconds'.format(time.time()-response_spectral_time))
                
            # Compute the reference power spectra
            reference_spectral_time = time.time()
            if self.spectral_processing_parameters.requires_full_spectral_reference:
                self.log('Computing Full Spectral Reference Matrix')
                self.reference_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.reference_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.reference_fft[~exclude_averages].shape[0]
                # Get the diagonal matrix as well
                self.reference_diagonal_matrix = np.einsum(
                    'fii->fi',
                    self.reference_spectral_matrix)
            elif self.spectral_processing_parameters.requires_diagonal_spectral_reference:
                self.log('Computing Diagonal of Spectral Reference Matrix')
                self.reference_diagonal_matrix = np.einsum(
                    'aif,aif->fi',
                    self.reference_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.reference_fft[~exclude_averages].shape[0]
            if (self.spectral_processing_parameters.requires_full_spectral_reference
                or self.spectral_processing_parameters.requires_diagonal_spectral_reference):
                self.log('Computed Reference Spectral Matrix in {:0.2f} seconds'.format(time.time()-reference_spectral_time))
                    
            # Compute cross spectra between reference and response
            if self.spectral_processing_parameters.requires_spectral_reference_response:
                cross_spectral_time = time.time()
                self.log('Computing Full Cross Spectral Response/Reference Matrix')
                self.response_reference_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.response_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.response_fft[~exclude_averages].shape[0]
                self.log('Computed Crossspectral Matrix in {:0.2f} seconds'.format(time.time()-cross_spectral_time))
            frames = self.spectral_processing_parameters.averages - np.sum(exclude_averages)
                
        else: # For exponential averaging
            for frame in data:
                response_fft, reference_fft = frame
                
                # Compute response spectra
                response_spectral_time = time.time()
                if self.spectral_processing_parameters.requires_full_spectral_response:
                    self.log('Computing Full Spectral Response Matrix')
                    if self.response_spectral_matrix is None:
                        self.response_spectral_matrix = np.einsum('if,jf->fij',response_fft,np.conj(response_fft))
                    else:
                        self.response_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',response_fft,np.conj(response_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_spectral_matrix
                            )
                    # Get the diagonal matrix as well
                    self.response_diagonal_matrix = np.einsum(
                        'fii->fi',
                        self.response_spectral_matrix)
                elif self.spectral_processing_parameters.requires_diagonal_spectral_response:
                    self.log('Computing Diagonal of Spectral Response Matrix')
                    if self.response_diagonal_matrix is None:
                        self.response_diagonal_matrix = np.einsum('if,if->fi',response_fft,np.conj(response_fft))
                    else:
                        self.response_diagonal_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,if->fi',response_fft,np.conj(response_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_diagonal_matrix
                            )
                if (self.spectral_processing_parameters.requires_full_spectral_response
                    or self.spectral_processing_parameters.requires_diagonal_spectral_response):
                    self.log('Computed Response Spectral Matrix in {:0.2f} seconds'.format(time.time()-response_spectral_time))
                        
                # Compute the reference spectra
                reference_spectral_time = time.time()
                if self.spectral_processing_parameters.requires_full_spectral_reference:
                    self.log('Computing Full Spectral Reference Matrix')
                    if self.reference_spectral_matrix is None:
                        self.reference_spectral_matrix = np.einsum('if,jf->fij',reference_fft,np.conj(reference_fft))
                    else:
                        self.reference_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',reference_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.reference_spectral_matrix
                            )
                    # Get the diagonal matrix as well
                    self.reference_diagonal_matrix = np.einsum(
                        'fii->fi',
                        self.reference_spectral_matrix)
                elif self.spectral_processing_parameters.requires_diagonal_spectral_reference:
                    self.log('Computing Diagonal of Spectral Reference Matrix')
                    if self.reference_diagonal_matrix is None:
                        self.reference_diagonal_matrix = np.einsum('if,if->fi',reference_fft,np.conj(reference_fft))
                    else:
                        self.reference_diagonal_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,if->fi',reference_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.reference_diagonal_matrix
                            )
                if (self.spectral_processing_parameters.requires_full_spectral_reference
                    or self.spectral_processing_parameters.requires_diagonal_spectral_reference):
                    self.log('Computed Reference Spectral Matrix in {:0.2f} seconds'.format(time.time()-reference_spectral_time))
                        
                # Compute reference and response cross spectra
                if self.spectral_processing_parameters.requires_spectral_reference_response:
                    cross_spectral_time = time.time()
                    self.log('Computing Full Cross Spectral Response/Reference Matrix')
                    if self.response_reference_spectral_matrix is None:
                        self.response_reference_spectral_matrix = np.einsum('if,jf->fij',response_fft,np.conj(reference_fft))
                    else:
                        self.response_reference_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',response_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_reference_spectral_matrix
                            )
                    self.log('Computed Crossspectral Matrix in {:0.2f} seconds'.format(time.time()-cross_spectral_time))
                self.frames_computed += 1
                
            frames = self.frames_computed
        self.log('Computed Spectral Matrices for {:} frames in {:0.2f} seconds'.format(frames, time.time() - response_spectral_time))
        Gffpinv = None
        Gfxpinv = None
        if self.spectral_processing_parameters.compute_frf:
            frf_time = time.time()
            if self.spectral_processing_parameters.frf_estimator == Estimator.H1:
                if Gffpinv is None:
                    Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
                frf = self.response_reference_spectral_matrix@Gffpinv
            elif self.spectral_processing_parameters.frf_estimator == Estimator.H2:
                Gfx = self.response_reference_spectral_matrix.conj().transpose(0,2,1)
                Gfxpinv = np.linalg.pinv(Gfx,rcond=1e-12,hermitian=True)
                frf = self.response_spectral_matrix@Gfxpinv
            elif self.spectral_processing_parameters.frf_estimator == Estimator.H3:
                if Gffpinv is None:
                    Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
                Gfx = self.response_reference_spectral_matrix.conj().transpose(0,2,1)
                if Gfxpinv is None:
                    Gfxpinv = np.linalg.pinv(Gfx,rcond=1e-12,hermitian=True)
                frf = (self.response_spectral_matrix@Gfxpinv + self.response_reference_spectral_matrix@Gffpinv)/2
            elif self.spectral_processing_parameters.frf_estimator == Estimator.HV:
                Gxx = self.response_diagonal_matrix.T[...,np.newaxis,np.newaxis]
                Gxf = np.einsum('fij->ifj',self.response_reference_spectral_matrix)[...,np.newaxis,:]
                Gff = self.reference_spectral_matrix
                Gff = np.broadcast_to(Gff,Gxx.shape[:-2]+Gff.shape[-2:])
                Gffx = np.block([[Gff, np.conj(np.moveaxis(Gxf, -2, -1))],
                                 [Gxf, Gxx]])
                # Compute eigenvalues
                lam, evect = np.linalg.eigh(np.moveaxis(Gffx, -2, -1))
                # Get the evect corresponding to the minimum eigenvalue
                evect = evect[..., 0]  # Assumes evals are sorted ascending
                frf = np.moveaxis(-evect[..., :-1] / evect[..., -1:],  # Scale so last value is -1
                                  -3, -2)
            self.log('Computed FRF in {:0.2f} seconds'.format(time.time()-frf_time))
            cond_time = time.time()
            frf_condition = np.linalg.cond(frf)
            self.log('Computed FRF Condition Number in {:0.2f} seconds'.format(time.time()-cond_time))
        else:
            frf = None
            frf_condition = None
        if self.spectral_processing_parameters.compute_coherence:
            coh_time = time.time()
            if Gffpinv is None:
                Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
            coherence = (np.einsum('fij,fjk,fik->fi',
                                   self.response_reference_spectral_matrix,
                                   Gffpinv,
                                   self.response_reference_spectral_matrix.conj()) / 
                         self.response_diagonal_matrix).real
            self.log('Computed Coherence in {:0.2f} seconds'.format(time.time()-coh_time))
        else:
            coherence = None
        if self.spectral_processing_parameters.compute_cpsd:
            cpsd_time = time.time()
            reference_spectral_matrix = self.reference_spectral_matrix.copy()
            response_spectral_matrix = self.response_spectral_matrix.copy()
            # Normalize
            response_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/ # Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            response_spectral_matrix[1:-1] *= 2
            reference_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/# Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            reference_spectral_matrix[1:-1] *= 2
            self.log('Computed CPSDs in {:0.2f} seconds'.format(time.time()-cpsd_time))
        elif self.spectral_processing_parameters.compute_apsd:
            apsd_time = time.time()
            reference_spectral_matrix = self.reference_diagonal_matrix.copy()
            response_spectral_matrix = self.response_diagonal_matrix.copy()
            # Normalize
            response_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/ # Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            response_spectral_matrix[1:-1] *= 2
            reference_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/# Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            reference_spectral_matrix[1:-1] *= 2
            self.log('Computed APSDs in {:0.2f} seconds'.format(time.time()-apsd_time))
        else:
            response_spectral_matrix = None
            reference_spectral_matrix = None
        frequencies = np.arange(self.spectral_processing_parameters.num_frequency_lines)*self.spectral_processing_parameters.frequency_spacing
        # --- Raw-FRF capture for offline verification (2026-09-03) ---
        # Companion to the CVA_CAPTURE_RAW_DATA hook above, for the H1/H2/
        # H3/HV path: dumps the just-computed FRF (whichever estimator is
        # selected) so it can be checked against ground truth offline,
        # the same way the CVA capture is checked -- e.g. to test whether
        # raising Integration Oversampling changes the *plant simulation's*
        # own fidelity (see design doc section 17), independent of which
        # FRF estimator is in use. Safe to leave on -- pure side file
        # write, no effect on control-path behavior.
        if H1_CAPTURE_FRF and frf is not None:
            try:
                capture_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'examples', 'sixdrive12resp', 'results', 'cva_captures')
                os.makedirs(capture_dir, exist_ok=True)
                np.savez(
                    os.path.join(capture_dir, 'latest_h1_sysid_capture.npz'),
                    frequencies=frequencies,
                    frf=frf,
                    estimator=self.spectral_processing_parameters.frf_estimator.name,
                    frames=frames,
                    sample_rate=self.spectral_processing_parameters.sample_rate,
                    response_reference_spectral_matrix=(
                        self.response_reference_spectral_matrix
                        if self.spectral_processing_parameters.requires_spectral_reference_response
                        else np.array([])),
                    reference_spectral_matrix=(
                        self.reference_spectral_matrix
                        if self.spectral_processing_parameters.requires_full_spectral_reference
                        else np.array([])),
                    capture_wall_time=time.time())
                self.log('H1/H2/H3/HV raw-FRF capture written to {:}'.format(capture_dir))
            except Exception as exc:
                self.log('H1 raw-FRF capture failed ({!r}); continuing without it'.format(exc))
        self.log('Sending Updated Spectral Data')
        self.data_out_queue.put((frames,frequencies,frf,coherence,
                                 response_spectral_matrix,
                                 reference_spectral_matrix,frf_condition))
        # Keep running
        self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))

    # ------------------------------------------------------------------
    # CVA-innovations estimator branch (design doc section 7 items 1/2/4).
    # Reached only when frf_estimator == Estimator.CVA_INNOVATIONS; the
    # H1/H2/H3/HV path above is completely unmodified otherwise.
    # ------------------------------------------------------------------
    def _load_cva_module(self):
        """Lazily load globalcva/global_cva_frf.py by path. globalcva/ isn't
        a package (no __init__.py) -- reuses load_python_module, the same
        runtime-loader utility already used elsewhere in this codebase for
        loading control laws by path, rather than inventing a new sys.path
        convention."""
        if self._cva_module is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._cva_module = load_python_module(
                os.path.join(repo_root,'globalcva','global_cva_frf.py'))
        return self._cva_module

    def _cva_explained_variance_coherence(self,frf):
        """Coherence-analog candidate (c) from cva_frf_integration_design's
        section 10 ("how much of the total response is explained by the FRF
        times the input?"): block-wise prediction residual vs actual
        response. Upper-bounded at 1 by construction for ANY H -- no
        per-bin-optimality assumption needed, unlike substituting H into
        Rattlesnake's H1-derived multiple-coherence formula above (section
        10 found that substitution unbounded even for the exact true
        system H, not just CVA's). Clipped to [0,1]; a negative raw value
        (model worse than predicting zero at that bin/channel) is
        legitimate signal, not a bug, but callers/GUI expect [0,1] like the
        H1/H2/H3/HV coherence field.

        Reuses the SAME nperseg as the live frequency grid (see
        _run_cva_processing) so frf -- already evaluated on that grid --
        can be used directly with no re-interpolation."""
        params = self.spectral_processing_parameters
        nperseg = 2*(params.num_frequency_lines-1)
        noverlap = nperseg//2
        win = sig.get_window('hann',nperseg)
        step = nperseg-noverlap
        n = self._cva_response_buffer.shape[-1]
        starts = list(range(0,n-nperseg+1,step))
        if not starts:
            return None
        num_reference = self._cva_reference_buffer.shape[0]
        num_response = self._cva_response_buffer.shape[0]
        Uf = np.zeros((len(starts),num_reference,nperseg//2+1),dtype=complex)
        Yf = np.zeros((len(starts),num_response,nperseg//2+1),dtype=complex)
        for bi,st in enumerate(starts):
            useg = self._cva_reference_buffer[:,st:st+nperseg]
            yseg = self._cva_response_buffer[:,st:st+nperseg]
            useg = useg-useg.mean(axis=1,keepdims=True)
            yseg = yseg-yseg.mean(axis=1,keepdims=True)
            Uf[bi] = np.fft.rfft(useg*win[np.newaxis,:],axis=1)
            Yf[bi] = np.fft.rfft(yseg*win[np.newaxis,:],axis=1)
        Yhat = np.einsum('fij,bjf->bif',frf,Uf)
        E = Yf-Yhat
        Gee = np.mean(np.abs(E)**2,axis=0)   # (num_response, F)
        Gyy = np.mean(np.abs(Yf)**2,axis=0)  # (num_response, F)
        Gyy_safe = np.where(Gyy == 0,1.0,Gyy)
        coh = np.clip(1.0-(Gee/Gyy_safe),0.0,1.0)
        return coh.T  # (F, num_response) -- matches the H1/H2/H3/HV coherence shape

    def _cva_diagonal_cpsd(self):
        """Lightweight per-channel (diagonal-only) Welch PSD of the raw CVA
        buffer, embedded into (F,M,M)/(F,N,N) matrices with zero off-
        diagonal cross-terms, purely so downstream code that expects the
        H1/H2/H3/HV full-matrix shape (GUI CPSD display, and any control
        law's sysid_response_cpsd/sysid_reference_cpsd argument -- see
        control_laws.py's match_coherence_phase) doesn't crash on a shape
        mismatch while CVA is selected. This is NOT the full cross-channel
        matrix H1/H2/H3/Hv produce -- known, documented simplification for
        this pass: a control law's buzz-baseline warm-start would see
        uncorrelated (diagonal-only) drive structure rather than the real
        measured cross-correlation, since off-diagonal terms are zero."""
        params = self.spectral_processing_parameters
        nperseg = 2*(params.num_frequency_lines-1)
        _,response_psd = sig.welch(self._cva_response_buffer,fs=params.sample_rate,
                                   nperseg=nperseg,axis=-1)
        _,reference_psd = sig.welch(self._cva_reference_buffer,fs=params.sample_rate,
                                    nperseg=nperseg,axis=-1)
        response_psd = response_psd.T   # (F, num_response)
        reference_psd = reference_psd.T # (F, num_reference)
        F = response_psd.shape[0]
        response_matrix = np.zeros((F,response_psd.shape[1],response_psd.shape[1]),dtype=complex)
        reference_matrix = np.zeros((F,reference_psd.shape[1],reference_psd.shape[1]),dtype=complex)
        ridx = np.arange(response_psd.shape[1])
        fidx = np.arange(reference_psd.shape[1])
        response_matrix[:,ridx,ridx] = response_psd
        reference_matrix[:,fidx,fidx] = reference_psd
        return response_matrix,reference_matrix

    def _run_cva_processing(self,data):
        """CVA-innovations estimator branch. Unlike H1/H2/H3/HV, which
        operate on windowed FFT frames accumulated into spectral matrices,
        CVA needs raw, un-windowed, causally-ordered time samples -- these
        arrive on self.raw_data_in_queue (the raw tap in data_collector.py's
        acquire(), gated by CollectorMetadata.raw_tap_enabled). Maintains a
        rolling window of raw samples, refits on a wall-clock throttle
        (cva_refit_interval_seconds -- a simple stand-in for real gating/
        publish logic, which is a separate, not-yet-built piece; see design
        doc section 7 item 5), and emits the SAME output tuple shape as the
        H1/H2/H3/HV path so RandomVibrationDataAnalysisProcess needs no
        changes (design doc section 7 item 2)."""
        params = self.spectral_processing_parameters
        if self.raw_data_in_queue is None:
            self.log('CVA_INNOVATIONS selected but raw_data_in_queue is not wired up -- '
                     'no raw data can arrive. Idling.')
            time.sleep(WAIT_TIME)
            self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
            return
        new_blocks = flush_queue(self.raw_data_in_queue,timeout=WAIT_TIME)
        if new_blocks:
            new_response = np.concatenate([b[0] for b in new_blocks],axis=-1)
            new_reference = np.concatenate([b[1] for b in new_blocks],axis=-1)
            if self._cva_response_buffer is None:
                self._cva_response_buffer = new_response
                self._cva_reference_buffer = new_reference
            else:
                self._cva_response_buffer = np.concatenate(
                    (self._cva_response_buffer,new_response),axis=-1)
                self._cva_reference_buffer = np.concatenate(
                    (self._cva_reference_buffer,new_reference),axis=-1)
            window_samples = int(params.cva_window_seconds*params.sample_rate)
            if self._cva_response_buffer.shape[-1] > window_samples:
                self._cva_response_buffer = self._cva_response_buffer[...,-window_samples:]
                self._cva_reference_buffer = self._cva_reference_buffer[...,-window_samples:]

        window_samples = int(params.cva_window_seconds*params.sample_rate)
        if (self._cva_response_buffer is None
                or self._cva_response_buffer.shape[-1] < window_samples):
            have = 0 if self._cva_response_buffer is None else self._cva_response_buffer.shape[-1]
            self.log('CVA buffer filling: {:}/{:} samples'.format(have,window_samples))
            time.sleep(WAIT_TIME)
            self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
            return

        now = time.time()
        if now-self._cva_last_fit_wall_time < params.cva_refit_interval_seconds:
            time.sleep(WAIT_TIME)
            self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
            return

        frequencies = np.arange(params.num_frequency_lines)*params.frequency_spacing
        dt = 1.0/params.sample_rate
        self._cva_last_fit_wall_time = now
        try:
            cva = self._load_cva_module()
            fit_time = time.time()
            result = cva.global_cva_innovations(
                self._cva_response_buffer,self._cva_reference_buffer,
                lags=params.cva_lags,tol=1e-10,rank=params.cva_rank,
                refine_iters=params.cva_refine_iters)
            frf = cva.frf_from_ss(result['A'],result['B'],result['C'],result['D'],frequencies,dt)
            self.log('Computed CVA FRF in {:0.2f} seconds ({:} samples, lags={:}, rank={:})'.format(
                time.time()-fit_time,self._cva_response_buffer.shape[-1],params.cva_lags,params.cva_rank))
            self._cva_last_frf = frf
            self._cva_last_condition = np.linalg.cond(frf)
            # --- Raw-data capture for offline verification (2026-09-02) ---
            # Temporarily enabled to directly answer: is the LIVE CVA fit as
            # accurate as the offline-validated method, on the SAME raw data?
            # (design doc section 16's "band-limited excitation" root cause
            # was found to be wrong -- get_sysid_signal_generator() hardcodes
            # low/high_frequency_cutoff=None, i.e. the live sys-ID excitation
            # is already full-bandwidth regardless of the Specification File.
            # That reopens the question this capture is meant to settle.)
            # Overwrites a single file each successful fit, so once sys-ID
            # completes it holds the LAST (most-averaged, about-to-be-used-
            # for-control) fit's raw data + resulting FRF. Safe to leave in
            # place -- pure side file write, no effect on control-path
            # behavior -- but flip CVA_CAPTURE_RAW_DATA to False once this
            # investigation is done.
            if CVA_CAPTURE_RAW_DATA:
                try:
                    capture_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'examples', 'sixdrive12resp', 'results', 'cva_captures')
                    os.makedirs(capture_dir, exist_ok=True)
                    np.savez(
                        os.path.join(capture_dir, 'latest_cva_sysid_capture.npz'),
                        response_buffer=self._cva_response_buffer,
                        reference_buffer=self._cva_reference_buffer,
                        frequencies=frequencies,
                        frf=frf,
                        A=result['A'], B=result['B'], C=result['C'], D=result['D'],
                        sample_rate=params.sample_rate,
                        cva_lags=params.cva_lags,
                        cva_rank=params.cva_rank,
                        cva_refine_iters=params.cva_refine_iters,
                        cva_window_seconds=params.cva_window_seconds,
                        capture_wall_time=now)
                    self.log('CVA raw-data capture written to {:}'.format(capture_dir))
                except Exception as exc:
                    self.log('CVA raw-data capture failed ({!r}); continuing without it'.format(exc))
            if params.compute_coherence:
                try:
                    self._cva_last_coherence = self._cva_explained_variance_coherence(frf)
                except Exception as exc:
                    self.log('CVA coherence-analog failed ({!r}); publishing FRF without coherence'.format(exc))
                    self._cva_last_coherence = None
        except Exception as exc:
            # EXPECTED and harmless during the zero-drive noise-floor phase --
            # CVA's Hankel/covariance matrices are exactly singular with no
            # persistent excitation on the reference channels (H1's pinv
            # degrades gracefully there instead of raising). Still fall
            # through to the emit below with the LAST successful frf/
            # coherence/condition (None on the very first failure, which is
            # fine -- run_sysid_noise() doesn't use them) -- emitting nothing
            # here would leave self.frames stuck at 0 in
            # AbstractSysIDAnalysisProcess forever, since that only advances
            # when spectral_data actually arrives (bug found live 2026-08-29:
            # noise-floor phase hung indefinitely, Start stayed grayed out).
            self.log('CVA fit failed ({!r}); publishing last known-good FRF/coherence '
                     '(None if none yet) so frame-count-driven phase transitions '
                     '(e.g. noise-floor completion) are not blocked'.format(exc))

        response_spectral_matrix = None
        reference_spectral_matrix = None
        if params.compute_cpsd or params.compute_apsd:
            try:
                response_spectral_matrix,reference_spectral_matrix = self._cva_diagonal_cpsd()
            except Exception as exc:
                self.log('CVA CPSD/APSD display computation failed ({!r})'.format(exc))

        frf = self._cva_last_frf
        coherence = self._cva_last_coherence
        frf_condition = self._cva_last_condition
        # H1/H2/H3/HV report "frames" as the number of averages actually
        # accumulated so far, and AbstractSysIDAnalysisProcess.run_sysid_noise/
        # run_sysid_transfer_function gate phase completion on an EXACT match
        # against the configured target (sysid_noise_averages/sysid_averages/
        # frames_in_cpsd, all of which flow into params.averages -- see
        # get_sysid_spectral_processing_metadata/get_spectral_processing_metadata).
        # CVA doesn't accumulate an incremental average the same way -- each
        # completed window IS the estimate -- so report the target directly
        # once we've reached one full window+throttle cycle; using the raw
        # sample count here (an earlier version of this method did) could
        # never satisfy that exact-equality check and would hang the same way
        # the missing-emit bug above did.
        frames = params.averages
        self.log('Sending Updated CVA Spectral Data')
        self.data_out_queue.put((frames,frequencies,frf,coherence,
                                 response_spectral_matrix,
                                 reference_spectral_matrix,frf_condition))
        self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))

    def clear_spectral_processing(self,data):
        """Clears all data in the buffer so the FRF starts fresh from new data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.frames_computed = 0
        self.response_spectral_matrix = None
        self.reference_spectral_matrix = None
        self.response_reference_spectral_matrix = None
        self._cva_response_buffer = None
        self._cva_reference_buffer = None
        self._cva_last_fit_wall_time = 0.0
        self._cva_last_frf = None
        self._cva_last_coherence = None
        self._cva_last_condition = None
        if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
            self.response_fft[:] = np.nan
            self.reference_fft[:] = np.nan
        else:
            self.response_fft = None
            self.reference_fft = None
    
    def stop_spectral_processing(self,data):
        """Stops computing FRFs from time data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        time.sleep(WAIT_TIME)
        flushed_data = self.command_queue.flush(self.process_name)
        # Put back any quit message that may have been pulled off
        for message,data in flushed_data:
            if message == GlobalCommands.QUIT:
                self.command_queue.put(self.process_name,(message,data))
        flush_queue(self.data_out_queue)
        self.environment_command_queue.put(self.process_name,(SpectralProcessingCommands.SHUTDOWN_ACHIEVED,None))

def spectral_processing_process(environment_name : str,
                               command_queue : VerboseMessageQueue,
                               data_in_queue : mp.queues.Queue,
                               data_out_queue : mp.queues.Queue,
                               environment_command_queue : VerboseMessageQueue,
                               gui_update_queue : mp.queues.Queue,
                               log_file_queue : mp.queues.Queue,
                               process_name = None,
                               raw_data_in_queue : mp.queues.Queue = None
                               ):
    """Function passed to multiprocessing as the FRF computation process
    
    This process creates the ``FRFComputationProcess`` object and calls the
    ``run`` function.


    Parameters
    ----------
    environment_name : str :
        Name of the environment that this subprocess belongs to.
    command_queue : VerboseMessageQueue :
        The queue containing instructions for the FRF process
    data_for_frf_queue : mp.queues.Queue :
        Queue containing input data for the FRF computation
    updated_frf_queue : mp.queues.Queue :
        Queue where frf process will put computed frfs
    gui_update_queue : mp.queues.Queue :
        Queue for gui updates
    log_file_queue : mp.queues.Queue :
        Queue for writing to the log file

    """
    
    spectral_processing_instance = SpectralProcessingProcess(
        environment_name + ' Spectral Processing Computation'
        if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        gui_update_queue,
        log_file_queue, environment_name,
        raw_data_in_queue = raw_data_in_queue)
    spectral_processing_instance.run()
