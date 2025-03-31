(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star4)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 planet5)
	(have_image planet5 image0)
))

)
