(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	spectrograph0 - mode
	image1 - mode
	thermograph2 - mode
	groundstation5 - direction
	star2 - direction
	star0 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite2 phenomenon8)
	(have_image star6 thermograph2)
	(have_image phenomenon7 thermograph2)
	(have_image phenomenon8 image1)
	(have_image planet9 spectrograph0)
))

)
