(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	thermograph0 - mode
	groundstation1 - direction
	star3 - direction
	groundstation0 - direction
	star2 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star3)
	(have_image phenomenon4 thermograph0)
	(have_image star5 thermograph0)
))

)
