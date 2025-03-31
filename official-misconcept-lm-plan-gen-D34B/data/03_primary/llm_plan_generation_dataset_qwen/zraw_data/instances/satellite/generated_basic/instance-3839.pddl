(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	thermograph0 - mode
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image star5 thermograph0)
	(have_image phenomenon6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
))

)
