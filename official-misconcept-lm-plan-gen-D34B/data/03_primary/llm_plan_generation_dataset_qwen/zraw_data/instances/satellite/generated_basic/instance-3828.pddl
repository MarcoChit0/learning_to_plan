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
	star4 - direction
	star5 - direction
	star1 - direction
	groundstation0 - direction
	star3 - direction
	star2 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
