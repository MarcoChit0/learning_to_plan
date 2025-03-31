(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	thermograph1 - mode
	thermograph0 - mode
	star3 - direction
	star2 - direction
	groundstation0 - direction
	star1 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
)
(:goal (and
	(have_image star4 thermograph0)
	(have_image phenomenon5 thermograph1)
	(have_image star6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image phenomenon8 thermograph0)
	(have_image planet9 thermograph1)
))

)
