(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	phenomenon3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite2 phenomenon3)
	(have_image phenomenon3 thermograph0)
	(have_image star4 thermograph0)
))

)
