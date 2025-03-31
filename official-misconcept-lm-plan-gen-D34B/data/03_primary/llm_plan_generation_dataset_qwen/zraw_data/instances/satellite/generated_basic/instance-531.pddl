(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image3 - mode
	thermograph2 - mode
	infrared0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared0)
	(supports instrument1 image3)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph1)
	(supports instrument2 image3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph2)
	(supports instrument4 image3)
	(calibration_target instrument4 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image star5 spectrograph1)
	(have_image star6 image3)
	(have_image star7 infrared0)
	(have_image phenomenon8 thermograph2)
))

)
