(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph1 - mode
	thermograph0 - mode
	groundstation1 - direction
	star2 - direction
	star3 - direction
	groundstation4 - direction
	groundstation0 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(have_image planet5 thermograph1)
	(have_image phenomenon6 thermograph1)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
))

)
