(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	star0 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image phenomenon4 image1)
	(have_image star5 thermograph0)
	(have_image planet6 thermograph0)
	(have_image planet7 image1)
	(have_image star8 thermograph0)
	(have_image star9 thermograph0)
))

)
