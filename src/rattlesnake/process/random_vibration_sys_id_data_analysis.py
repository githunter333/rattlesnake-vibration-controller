# -*- coding: utf-8 -*-
"""
This file defines the data analysis and control law for the Random Vibration
Environment

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
import time
from enum import Enum

import numpy as np
import netCDF4 as nc4

from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIDAnalysisProcess,
    SysIdDataAnalysisCommands,
    SysIdMetadata,
)
from rattlesnake.environment.abstract_interactive_control_law import ControlLawCommands
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    flush_queue,
    power2db,
    rms_csd,
    rms_time,
    load_control_module,
)
from rattlesnake.user_interface.ui_utilities import UICommands


# region Commands
class RandomVibrationDataAnalysisCommands(Enum):
    """Enumeration containing valid commands for the random data analysis process"""

    INITIALIZE_ENVIRONMENT = 0
    INITIALIZE_PARAMETERS = 1
    PERFORM_CONTROL_PREDICTION = 2
    RUN_CONTROL = 3
    STOP_CONTROL = 4
    SAVE_CONTROL_DATA = 5
    SET_TEST_LEVEL_DB = 6
    # Sent to the environment, which owns the spectral command queue, to ask
    # spectral processing to put the full response CPSD on the next payload.
    REQUEST_FULL_RESPONSE_CPSD = 7


class RandomVibrationDataAnalysisUICommands(Enum):
    INTERACTIVE_CONTROL_SYSID_UPDATE = 0
    CONTROL_PREDICTIONS = 1
    CONTROL_UPDATE = 2
    UPDATE_TEST_RESPONSE_ERROR_LIST = 3


# endregion


# region Data Analysis
class RandomVibrationDataAnalysisProcess(SysIDAnalysisProcess):
    """Control calculations for the Random Vibration environment"""

    def __init__(
        self,
        process_name: str,
        command_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        environment_command_queue: VerboseMessageQueue,
        log_file_queue: mp.queues.Queue,
        gui_update_queue: mp.queues.Queue,
        environment_name: str,
        ping_alive_event: mp.synchronize.Event,
    ):
        super().__init__(
            process_name,
            command_queue,
            data_in_queue,
            data_out_queue,
            environment_command_queue,
            log_file_queue,
            gui_update_queue,
            environment_name,
            ping_alive_event,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.INITIALIZE_ENVIRONMENT,
            self.initialize_environment,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
            self.initialize_sysid_parameters,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION,
            self.perform_control_prediction,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.RUN_CONTROL, self.run_control
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.STOP_CONTROL, self.stop_control
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.SAVE_CONTROL_DATA,
            self.save_spectral_data,
        )
        self.map_command(
            RandomVibrationDataAnalysisCommands.SET_TEST_LEVEL_DB,
            self.set_test_level_db,
        )
        self.map_command(
            ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(
            ControlLawCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command
        )
        self.error_indices = None
        self.control_function = None
        self.response_cpsd_prediction = None
        self.drive_cpsd_prediction = None
        self.control_frf = None
        self.control_coherence = None
        self.control_frf_condition = None
        self.last_response_cpsd = None
        # True when last_response_cpsd carries real off-diagonals.  In
        # diagonal-only transport the off-diagonals arrive as zeros, and the
        # .nc4 save must not write those as though they were measured.
        self.last_response_cpsd_is_full = False
        # Filename of a save waiting for a full response CPSD to arrive.
        self.pending_spectral_save = None
        self.last_drive_cpsd = None
        self.startup = True
        self.has_sent_interactive_control_transfer_function_results = False
        self.last_interactive_parameters = None
        self.current_test_level_db = None

    # region StateSync
    def initialize_sysid_parameters(self, data: SysIdMetadata):
        self.parameters: SysIdMetadata
        super().initialize_sysid_parameters(data)  # This defines self.parameters

        # Find the frequency lines to perform control and compute error over
        # print(type(data.specification_cpsd_matrix))
        # print(data.specification_cpsd_matrix)
        self.error_indices = ~np.all(
            (self.environment_metadata.specification_cpsd_matrix == 0)
            | np.isnan(self.environment_metadata.specification_cpsd_matrix),
            axis=(-1, -2),
        )
        self.frequencies = self.environment_metadata.frequency_spacing * np.arange(
            self.environment_metadata.fft_lines
        )
        # Load in the control script
        module = load_control_module(self.environment_metadata.control_python_script)
        # Pull out the function from the loaded module
        if self.environment_metadata.control_python_function_type == 1:  # Generator
            # Get the generator function
            generator_function = getattr(
                module, self.environment_metadata.control_python_function
            )()
            # Get us to the first yield statement
            next(generator_function)
            # Define the control function as the generator's send function
            self.control_function = generator_function.send
        elif self.environment_metadata.control_python_function_type == 2:  # Class
            control_frf = (
                self.control_frf
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_coherence
            )
            self.control_function = getattr(
                module, self.environment_metadata.control_python_function
            )(
                self.environment_metadata.specification_cpsd_matrix,  # Specifications
                self.environment_metadata.specification_warning_matrix,  # Warning levels
                self.environment_metadata.specification_abort_matrix,  # Abort Levels
                self.environment_metadata.control_python_function_parameters,  # Extra parameters for the control law
                control_frf,  # Transfer Functions
                self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                self.sysid_data.sysid_reference_noise,  # from the system identification
                self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_data.sysid_reference_cpsd,  # from the system identification
                control_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD
                self.environment_metadata.frames_in_cpsd,  # Total number of frames
                self.last_response_cpsd,  # Last Control Response for Error Correction
                self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
            )
        elif self.environment_metadata.control_python_function_type == 3:  # Interactive
            control_frf = (
                self.control_frf
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_coherence
            )
            control_class = getattr(
                module, self.environment_metadata.control_python_function
            )
            self.control_function = control_class(
                self.environment_name,
                self.gui_update_queue,
                self.environment_metadata.specification_cpsd_matrix,  # Specifications
                self.environment_metadata.specification_warning_matrix,  # Warning levels
                self.environment_metadata.specification_abort_matrix,  # Abort Levels
                self.environment_metadata.control_python_function_parameters,  # Extra parameters for the control law
                control_frf,  # Transfer Functions
                self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                self.sysid_data.sysid_reference_noise,  # from the system identification
                self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_data.sysid_reference_cpsd,  # from the system identification
                control_coherence,  # Coherence from the system identification
                self.frames,  # Number of frames in the CPSD
                self.environment_metadata.frames_in_cpsd,  # Total number of frames
                self.last_response_cpsd,  # Last Control Response for Error Correction
                self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
            )
            self.last_interactive_parameters = None
            self.has_sent_interactive_control_transfer_function_results = False
        else:  # Function
            self.control_function = getattr(
                module, self.environment_metadata.control_python_function
            )
        # If a test level arrived before the control law was built, hand it
        # over now.  No-op unless the law defines set_test_level_db.
        self._forward_test_level_db()

    def _forward_test_level_db(self):
        """Pass the current test level (dB) to a class-style control law that
        defines set_test_level_db.  Laws that don't care never see it."""
        if (
            self.current_test_level_db is not None
            and self.control_function is not None
            and hasattr(self.control_function, "set_test_level_db")
        ):
            self.control_function.set_test_level_db(self.current_test_level_db)

    def set_test_level_db(self, data):
        """Receives the current test level in dB and forwards it to the control law"""
        self.current_test_level_db = data
        self._forward_test_level_db()

    # endregion

    # region Commands
    def perform_control_prediction(self, data):  # pylint: disable=unused-argument
        """Runs the control law with system identification information to predict control"""
        if self.sysid_data.sysid_frf is None:
            return
        if self.environment_metadata.control_python_function_type == 1:  # Generator
            output_cpsd = self.control_function(
                (
                    self.environment_metadata.specification_cpsd_matrix,  # Specifications
                    self.environment_metadata.specification_warning_matrix,  # Warning levels
                    self.environment_metadata.specification_abort_matrix,  # Abort Levels
                    self.sysid_data.sysid_frf,  # Transfer Functions
                    self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_data.sysid_reference_noise,  # from the system identification
                    self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_data.sysid_reference_cpsd,  # from the system identification
                    self.sysid_data.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                    self.environment_metadata.control_python_function_parameters,  # Extra parameters
                    None,  # Last Control Response for Error Correction
                    None,  # Last Control Excitation for Drive-based control
                )
            )
        elif self.environment_metadata.control_python_function_type in [
            2,
            3,
        ]:  # Class or Interactive
            if (
                self.environment_metadata.control_python_function == 2
                or not self.has_sent_interactive_control_transfer_function_results
            ):
                self.control_function.system_id_update(
                    self.sysid_data.sysid_frf,  # Transfer Functions
                    self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_data.sysid_reference_noise,  # from the system identification
                    self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_data.sysid_reference_cpsd,  # from the system identification
                    self.sysid_data.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                )
                if self.environment_metadata.control_python_function_type == 3:
                    self.gui_update_queue.put(
                        (
                            self.environment_name,
                            (
                                RandomVibrationDataAnalysisUICommands.INTERACTIVE_CONTROL_SYSID_UPDATE,
                                (
                                    self.sysid_data.sysid_frf,  # Transfer Functions
                                    self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                                    self.sysid_data.sysid_reference_noise,  # from the system identification
                                    self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                                    self.sysid_data.sysid_reference_cpsd,  # from the system identification
                                    self.sysid_data.sysid_coherence,  # Coherence from the system id
                                ),
                            ),
                        )
                    )
                    self.has_sent_interactive_control_transfer_function_results = True
            if (
                self.environment_metadata.control_python_function_type == 2
                or self.last_interactive_parameters is not None
            ):
                output_cpsd = self.control_function.control(
                    self.sysid_data.sysid_frf,  # Transfer Functions
                    self.sysid_data.sysid_coherence,  # Coherence from the system identification
                    self.parameters.sysid_averages,  # Number of frames in the CPSD
                    self.parameters.sysid_averages,  # Total number of frames
                    None,
                    None,
                )
            else:
                self.log(
                    "Have not yet received control parameters from interactive control law!"
                )
                self.command_queue.put(
                    self.process_name,
                    (
                        RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION,
                        None,
                    ),
                )
                time.sleep(0.25)
                return
        else:  # Function
            output_cpsd = self.control_function(
                self.environment_metadata.specification_cpsd_matrix,  # Specifications
                self.environment_metadata.specification_warning_matrix,  # Warning levels
                self.environment_metadata.specification_abort_matrix,  # Abort Levels
                self.sysid_data.sysid_frf,  # Transfer Functions
                self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                self.sysid_data.sysid_reference_noise,  # from the system identification
                self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                self.sysid_data.sysid_reference_cpsd,  # from the system identification
                self.sysid_data.sysid_coherence,  # Coherence from the system identification
                self.parameters.sysid_averages,  # Number of frames in the CPSD
                self.parameters.sysid_averages,  # Total number of frames
                self.environment_metadata.control_python_function_parameters,  # Extra parameters
                None,  # Last Control Response for Error Correction
                None,  # Last Control Excitation for Drive-based control
            )
        response_cpsd = (
            self.sysid_data.sysid_frf
            @ output_cpsd
            @ self.sysid_data.sysid_frf.conjugate().transpose(0, 2, 1)
        )
        self.drive_cpsd_prediction = output_cpsd
        self.response_cpsd_prediction = response_cpsd
        rms_drives = rms_csd(
            self.drive_cpsd_prediction, self.environment_metadata.frequency_spacing
        )
        response_db_error = power2db(
            np.einsum("ijj->ij", self.response_cpsd_prediction[self.error_indices]).real
        ) - power2db(
            np.einsum(
                "ijj->ij",
                self.environment_metadata.specification_cpsd_matrix[self.error_indices],
            ).real
        )
        rms_db_error = rms_time(response_db_error, axis=0)
        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    RandomVibrationDataAnalysisUICommands.CONTROL_PREDICTIONS,
                    (
                        self.frequencies,
                        self.drive_cpsd_prediction,
                        self.response_cpsd_prediction,
                        self.environment_metadata.specification_cpsd_matrix,
                        rms_drives,
                        rms_db_error,
                    ),
                ),
            )
        )
        if self.environment_metadata.control_python_function_type == 3:
            self.has_sent_interactive_control_transfer_function_results = False

    @staticmethod
    def _expand_response_cpsd(response):
        """Puts a transported response CPSD diagonal back into a full matrix.

        Spectral processing may send only the diagonal of the response CPSD,
        which is all any control law reads.  The laws are still handed a full
        (lines, channels, channels) matrix, because that is the published
        interface -- they slice it themselves with ``einsum('fmm->fm')``,
        ``np.diagonal`` or ``[:, i, i]`` -- so the diagonal goes back onto a
        full matrix here.  The off-diagonals are zero, not measured.

        A fresh array is allocated on every update rather than one buffer
        being reused, because laws are allowed to keep the array they are
        given: control_laws.py stores it in ``runtime_symbols`` and reads it on
        a later call, and a reused buffer would change under it.  That costs
        about 27 ms for a 47-channel, 8193-line matrix -- writing a diagonal
        touches one element per page, so the whole allocation does get faulted
        in -- against roughly 3.1 s saved by not moving the full matrix
        between processes.  Reusing a buffer would save that 27 ms and risk
        corrupting a law that kept its array, which is the wrong trade.
        """
        if response is None or response.ndim == 3:
            return response
        num_lines, num_channels = response.shape
        full = np.zeros(
            (num_lines, num_channels, num_channels), dtype=response.dtype
        )
        # Writable strided view of the diagonals; einsum would return a copy.
        full.reshape(num_lines, num_channels * num_channels)[
            :, :: num_channels + 1
        ] = response
        return full

    def run_control(self, data):  # pylint: disable=unused-argument
        """Runs the control law to generate new output CPSDs"""
        if self.startup:
            self.log("Starting Control")
            self.frames = 0
            self.data_out_queue.put([self.drive_cpsd_prediction])
            self.startup = False
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.log("Obtained Spectral Data")
            (
                self.frames,
                self.frequencies,
                self.control_frf,
                self.control_coherence,
                self.last_response_cpsd,
                self.last_drive_cpsd,
                self.control_frf_condition,
            ) = spectral_data[-1]
            # What arrived: the full matrix, or just its diagonal.  The
            # compact form is what goes to the GUI; the laws get the expanded
            # one.  Keep both so neither pays for the other.
            transported_response_cpsd = self.last_response_cpsd
            self.last_response_cpsd_is_full = (
                self.last_response_cpsd is not None
                and self.last_response_cpsd.ndim == 3
            )
            self.last_response_cpsd = self._expand_response_cpsd(
                self.last_response_cpsd
            )
            # Check to see if there are any aborts or warnings
            warning_channels = []
            abort_channels = []
            with np.errstate(invalid="ignore"):
                lines_out = (
                    self.environment_metadata.percent_lines_out / 100
                ) * self.environment_metadata.fft_lines
                for i in range(self.last_response_cpsd.shape[-1]):
                    if (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            > self.environment_metadata.specification_abort_matrix[
                                1, :, i
                            ]
                        )
                        > lines_out
                    ):
                        abort_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            < self.environment_metadata.specification_abort_matrix[
                                0, :, i
                            ]
                        )
                        > lines_out
                    ):
                        abort_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            > self.environment_metadata.specification_warning_matrix[
                                1, :, i
                            ]
                        )
                        > lines_out
                    ):
                        warning_channels.append(i)
                    elif (
                        sum(
                            self.last_response_cpsd[:, i, i]
                            < self.environment_metadata.specification_warning_matrix[
                                0, :, i
                            ]
                        )
                        > lines_out
                    ):
                        warning_channels.append(i)
            # The frame count must be >=, not ==, so that the abort stays
            # armed for the whole run.  With linear averaging the count
            # saturates at frames_in_cpsd, so == held forever and >= is
            # identical.  With exponential averaging the count keeps climbing,
            # so == would arm the abort for a single update and then disable it
            # for the rest of the test.
            if (
                len(abort_channels) > 0
                and self.frames >= self.environment_metadata.frames_in_cpsd
                and self.environment_metadata.allow_automatic_aborts
            ):
                print(f"Aborting due to channel indices {abort_channels}")
                self.log(f"Aborting due to channel indices {abort_channels}")
                self.environment_command_queue.put(
                    self.process_name,
                    (RandomVibrationDataAnalysisCommands.STOP_CONTROL, None),
                )
            response_db_error = power2db(
                np.einsum("ijj->ij", self.last_response_cpsd[self.error_indices]).real
            ) - power2db(
                np.einsum(
                    "ijj->ij",
                    self.environment_metadata.specification_cpsd_matrix[
                        self.error_indices
                    ],
                ).real
            )
            rms_db_error = rms_time(response_db_error, axis=0)
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        RandomVibrationDataAnalysisUICommands.UPDATE_TEST_RESPONSE_ERROR_LIST,
                        (rms_db_error, warning_channels, abort_channels),
                    ),
                )
            )
            self.log("Controlling")
            # Create the new control output
            control_frf = (
                self.control_frf
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_frf
            )
            control_coherence = (
                self.control_coherence
                if self.environment_metadata.update_tf_during_control
                else self.sysid_data.sysid_coherence
            )
            if self.environment_metadata.control_python_function_type == 1:  # Generator
                output_cpsd = self.control_function(
                    (
                        self.environment_metadata.specification_cpsd_matrix,  # Specifications
                        self.environment_metadata.specification_warning_matrix,  # Warning levels
                        self.environment_metadata.specification_abort_matrix,  # Abort Levels
                        control_frf,  # Transfer Functions
                        self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                        self.sysid_data.sysid_reference_noise,  # from the system identification
                        self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                        self.sysid_data.sysid_reference_cpsd,  # from the system identification
                        control_coherence,  # Coherence
                        self.frames,  # Number of frames in the CPSD
                        self.environment_metadata.frames_in_cpsd,  # Total number of frames
                        self.environment_metadata.control_python_function_parameters,  # Extra parameters
                        self.last_response_cpsd,  # Last Control Response for Error Correction
                        self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                    )
                )
            elif self.environment_metadata.control_python_function_type in [
                2,
                3,
            ]:  # Class or interactive class
                output_cpsd = self.control_function.control(
                    control_frf,  # Transfer Functions
                    control_coherence,  # Coherence
                    self.frames,  # Number of frames in the CPSD
                    self.environment_metadata.frames_in_cpsd,  # Total number of frames
                    self.last_response_cpsd,  # Last Control Response for Error Correction
                    self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                )
            else:  # Function
                output_cpsd = self.control_function(
                    self.environment_metadata.specification_cpsd_matrix,  # Specifications
                    self.environment_metadata.specification_warning_matrix,  # Warning levels
                    self.environment_metadata.specification_abort_matrix,  # Abort Levels
                    control_frf,  # Transfer Functions
                    self.sysid_data.sysid_response_noise,  # Noise levels and correlation
                    self.sysid_data.sysid_reference_noise,  # from the system identification
                    self.sysid_data.sysid_response_cpsd,  # Response levels and correlation
                    self.sysid_data.sysid_reference_cpsd,  # from the system identification
                    control_coherence,  # Coherence
                    self.frames,  # Number of frames in the CPSD
                    self.environment_metadata.frames_in_cpsd,  # Total number of frames
                    self.environment_metadata.control_python_function_parameters,  # Extra parameters
                    self.last_response_cpsd,  # Last Control Response for Error Correction
                    self.last_drive_cpsd,  # Last Control Excitation for Drive-based control
                )
            self.log(
                f"RMS Outputs from Control \n  "
                f"{rms_csd(output_cpsd, self.environment_metadata.frequency_spacing)}"
            )
            self.data_out_queue.put([output_cpsd])
            self.log("Finished Controlling")
            # Sent after the drive, not before the law.  This payload is the
            # largest message the process forwards, and pickling it ahead of
            # the law put its whole cost on the path between a new spectral
            # estimate and the drive computed from it.  The GUI is a display
            # and does not care about the few milliseconds; the shaker does.
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        RandomVibrationDataAnalysisUICommands.CONTROL_UPDATE,
                        (
                            self.frames,
                            self.environment_metadata.frames_in_cpsd,
                            self.frequencies,
                            self.control_frf,
                            self.control_coherence,
                            transported_response_cpsd,
                            self.last_drive_cpsd,
                            self.control_frf_condition,
                        ),
                    ),
                )
            )
            # A save may be waiting for off-diagonals to arrive.
            if (
                self.pending_spectral_save is not None
                and self.last_response_cpsd_is_full
            ):
                filename = self.pending_spectral_save
                self.pending_spectral_save = None
                self.log("Full response CPSD arrived; completing deferred save")
                self._write_spectral_data(filename)
            rms_voltages = rms_csd(
                output_cpsd, self.environment_metadata.frequency_spacing
            )
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (UICommands.SET_ATTR, ("test_output_voltage_list", rms_voltages)),
                )
            )
        self.command_queue.put(
            self.process_name, (RandomVibrationDataAnalysisCommands.RUN_CONTROL, None)
        )

    def update_interactive_control_parameters(self, interactive_control_parameters):
        """Updates parameters for the interactive control law"""
        if self.environment_metadata.control_python_function_type == 3:  # Interactive
            self.control_function.update_parameters(interactive_control_parameters)
            self.last_interactive_parameters = interactive_control_parameters
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def send_interactive_command(self, command):
        """Sends a command to the interactive control law"""
        if self.environment_metadata.control_python_function_type == 3:  # Interactive
            self.control_function.send_command(command)
        else:
            raise ValueError(
                "Received an UPDATE_INTERACTIVE_CONTROL_PARAMETERS signal without an "
                "interactive control law.  How did this happen?"
            )

    def stop_control(self, data):  # pylint: disable=unused-argument
        """Stops the data acquisition process"""
        # Remove any run_transfer_function or run_control from the queue
        instructions = self.command_queue.flush(self.process_name)
        for instruction in instructions:
            if not instruction[0] in [RandomVibrationDataAnalysisCommands.RUN_CONTROL]:
                self.command_queue.put(self.process_name, instruction)
        flush_queue(self.data_out_queue)
        # A save still waiting on off-diagonals would wait forever once the
        # payloads stop, so write it now with what is in hand.  The file
        # records which it got.
        if self.pending_spectral_save is not None:
            filename = self.pending_spectral_save
            self.pending_spectral_save = None
            self.log("Control stopping with a save pending; writing it now")
            self._write_spectral_data(filename)
        self.startup = True
        self.control_frf = None
        self.control_coherence = None
        self.control_frf_condition = None
        self.last_response_cpsd = None
        self.last_response_cpsd_is_full = False
        self.last_drive_cpsd = None
        self.environment_command_queue.put(
            self.process_name, (SysIdDataAnalysisCommands.SHUTDOWN_ACHIEVED, None)
        )

    def save_spectral_data(self, data):
        """Saves the spectral data, fetching off-diagonals first if needed.

        Under diagonal-only transport the live response CPSD held here has
        zeros off its diagonal, and this file records the full matrix.  So ask
        spectral processing for the full matrix and let the save complete when
        the next payload brings it, which is within one spectral update.  A
        save requested while control is not running cannot wait for a payload
        that will never come, so it writes immediately and the file records
        that the off-diagonals were not transported.
        """
        live = self.last_response_cpsd is not None
        if live and not self.last_response_cpsd_is_full and not self.startup:
            self.pending_spectral_save = data
            self.environment_command_queue.put(
                self.process_name,
                (
                    RandomVibrationDataAnalysisCommands.REQUEST_FULL_RESPONSE_CPSD,
                    None,
                ),
            )
            self.log(
                "Deferring the save until the full response CPSD arrives"
            )
            return
        self._write_spectral_data(data)

    def _write_spectral_data(self, data):
        """Save the spectral data for this environment to `data` (a filename).

        Writes the LIVE CONTROL spectra once control has produced any, and the
        system-identification spectra before that.  `spectral_data_source` on
        the group says which, so a reader never has to guess.

        This used to write the system-ID arrays unconditionally, which made
        every save in a run identical regardless of test level, control law or
        elapsed time -- the system ID is measured once, before control.  The
        pre-reorganization code (components/random_vibration_sys_id_environment
        .py) wrote last_response_cpsd / last_transfer_function / last_coherence
        / last_reference_cpsd here, and the sixdrive12resp bounded-softening
        files scored in results/notes/moving_plant_scores.json hold live control
        data: their response spectra vary with level (sums 7.74 / 11.36 / 45.52)
        and reproduce the published scores to two decimals.  Restoring that is
        what this does.  See
        examples/sixdrive12resp/results/notes/nonlinear_harness_2026-11-05.md.

        The noise spectra have no control-time equivalent -- they are measured
        with the drives quiet -- so they are always the system ID's.
        """
        filename = data

        live = self.last_response_cpsd is not None
        frf = (
            self.control_frf
            if live and self.control_frf is not None
            else self.sysid_data.sysid_frf
        )
        coherence = (
            self.control_coherence
            if live and self.control_coherence is not None
            else self.sysid_data.sysid_coherence
        )
        response_cpsd = (
            self.last_response_cpsd if live else self.sysid_data.sysid_response_cpsd
        )
        drive_cpsd = (
            self.last_drive_cpsd
            if live and self.last_drive_cpsd is not None
            else self.sysid_data.sysid_reference_cpsd
        )

        netcdf_dataset = nc4.Dataset(filename, "a", format="NETCDF4")
        netcdf_handle = netcdf_dataset.groups[self.environment_name]
        netcdf_handle.spectral_data_source = "control" if live else "system_id"
        # Says whether the response CPSD off-diagonals in this file were
        # measured or are zeros left by diagonal-only transport, so a reader
        # never mistakes one for the other.
        netcdf_handle.response_cpsd_off_diagonals = (
            "measured"
            if (not live or self.last_response_cpsd_is_full)
            else "not_transported"
        )
        netcdf_handle.spectral_data_frames = int(self.frames) if live else 0
        netcdf_handle.createDimension("drive_channels", frf.shape[2])

        def put(name, dims, values):
            netcdf_handle.createVariable(name, "f8", dims)[...] = values

        spec_spec = ("fft_lines", "specification_channels", "specification_channels")
        spec_drive = ("fft_lines", "specification_channels", "drive_channels")
        drive_drive = ("fft_lines", "drive_channels", "drive_channels")
        sysid = self.sysid_data

        put("frf_data_real", spec_drive, frf.real)
        put("frf_data_imag", spec_drive, frf.imag)
        put("frf_coherence", ("fft_lines", "specification_channels"), coherence.real)
        put("response_cpsd_real", spec_spec, response_cpsd.real)
        put("response_cpsd_imag", spec_spec, response_cpsd.imag)
        put("drive_cpsd_real", drive_drive, drive_cpsd.real)
        put("drive_cpsd_imag", drive_drive, drive_cpsd.imag)
        # Noise is a drives-quiet measurement; there is no control-time version.
        put("response_noise_cpsd_real", spec_spec, sysid.sysid_response_noise.real)
        put("response_noise_cpsd_imag", spec_spec, sysid.sysid_response_noise.imag)
        put("drive_noise_cpsd_real", drive_drive, sysid.sysid_reference_noise.real)
        put("drive_noise_cpsd_imag", drive_drive, sysid.sysid_reference_noise.imag)

        netcdf_dataset.close()

    # endregion


# endregion


# region Process
def random_data_analysis_process(
    environment_name: str,
    command_queue: VerboseMessageQueue,
    data_in_queue: mp.queues.Queue,
    data_out_queue: mp.queues.Queue,
    environment_command_queue: VerboseMessageQueue,
    gui_update_queue: mp.queues.Queue,
    log_file_queue: mp.queues.Queue,
    ping_alive_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event = None,
    process_name=None,
):
    """Process defining the random vibration control calculations and data analysis

    Parameters
    ----------
    environment_name : str
        The name of the random vibration environment
    command_queue : VerboseMessageQueue
        A message queue that will provide commands to this process
    data_in_queue : mp.queues.Queue
        A queue from which data will be received to use for control updates
    data_out_queue : mp.queues.Queue
        A queue into which the results of control calculations will be placed
    environment_command_queue : VerboseMessageQueue
        A message queue to send commands back to the main controller
    gui_update_queue : mp.queues.Queue
        A queue to send updates to the graphical user interface for updating interactive control
        laws
    log_file_queue : mp.queues.Queue
        A queue to send messages that will be logged in the log file
    process_name : str, optional
        A name for the process.  If not specified, it will be
    """
    data_analysis_instance = RandomVibrationDataAnalysisProcess(
        environment_name + " Data Analysis" if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        environment_name,
        ping_alive_event,
    )

    data_analysis_instance.run(shutdown_event)


# endregion
