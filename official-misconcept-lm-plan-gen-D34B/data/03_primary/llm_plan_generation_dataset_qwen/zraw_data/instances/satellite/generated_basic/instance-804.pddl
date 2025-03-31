(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	infrared1 - mode
	thermograph0 - mode
	star1 - direction
	star2 - direction
	star4 - direction
	groundstation3 - direction
	star0 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image star5 infrared1)
	(have_image planet6 infrared1)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
))

)
