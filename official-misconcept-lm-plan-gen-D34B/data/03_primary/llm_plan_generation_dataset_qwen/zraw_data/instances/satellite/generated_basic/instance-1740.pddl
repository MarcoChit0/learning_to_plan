(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	groundstation2 - direction
	star3 - direction
	groundstation5 - direction
	star4 - direction
	star1 - direction
	star0 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
)
(:goal (and
	(pointing satellite0 phenomenon9)
	(have_image star6 thermograph2)
	(have_image phenomenon7 infrared0)
	(have_image phenomenon8 spectrograph1)
	(have_image phenomenon9 spectrograph1)
	(have_image star10 thermograph2)
))

)
