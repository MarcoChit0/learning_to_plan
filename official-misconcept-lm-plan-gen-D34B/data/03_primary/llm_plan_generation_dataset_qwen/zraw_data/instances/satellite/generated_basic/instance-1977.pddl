(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	thermograph0 - mode
	star3 - direction
	star2 - direction
	star1 - direction
	groundstation0 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star1)
	(have_image star4 thermograph0)
	(have_image phenomenon5 thermograph0)
))

)
