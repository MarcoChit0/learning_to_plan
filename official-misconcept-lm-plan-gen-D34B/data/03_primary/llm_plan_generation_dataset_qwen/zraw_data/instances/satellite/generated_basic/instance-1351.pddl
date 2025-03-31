(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	image3 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image3)
	(supports instrument7 image0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet1 image0)
))

)
