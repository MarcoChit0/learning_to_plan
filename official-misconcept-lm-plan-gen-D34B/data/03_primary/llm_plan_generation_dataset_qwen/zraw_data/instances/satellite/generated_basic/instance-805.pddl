(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph1 - mode
	image0 - mode
	groundstation3 - direction
	star4 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation1 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(have_image phenomenon5 spectrograph1)
	(have_image star6 spectrograph1)
	(have_image star7 spectrograph1)
	(have_image star8 spectrograph1)
))

)
