(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph2 - mode
	spectrograph1 - mode
	spectrograph3 - mode
	infrared0 - mode
	star0 - direction
	star3 - direction
	star4 - direction
	groundstation2 - direction
	groundstation1 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph3)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
)
(:goal (and
	(have_image phenomenon5 spectrograph3)
	(have_image star6 spectrograph3)
	(have_image planet7 spectrograph3)
	(have_image star8 infrared0)
))

)
