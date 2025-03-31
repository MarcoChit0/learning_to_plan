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
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon1)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 phenomenon2)
	(have_image phenomenon1 thermograph0)
	(have_image phenomenon2 thermograph0)
))

)
