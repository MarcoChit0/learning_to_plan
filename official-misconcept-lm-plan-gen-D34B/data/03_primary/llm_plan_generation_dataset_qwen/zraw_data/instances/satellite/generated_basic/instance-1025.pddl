(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image1 - mode
	thermograph2 - mode
	thermograph0 - mode
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image planet4 thermograph2)
	(have_image planet5 image1)
	(have_image phenomenon6 thermograph0)
	(have_image star7 image1)
	(have_image phenomenon8 thermograph0)
))

)
