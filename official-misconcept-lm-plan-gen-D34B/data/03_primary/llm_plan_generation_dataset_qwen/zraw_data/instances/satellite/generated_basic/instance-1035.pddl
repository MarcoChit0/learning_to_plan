(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	thermograph0 - mode
	image2 - mode
	thermograph1 - mode
	star3 - direction
	groundstation2 - direction
	star0 - direction
	star1 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph1)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 image2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
)
(:goal (and
	(pointing satellite0 phenomenon4)
	(have_image phenomenon4 image2)
	(have_image star5 thermograph0)
	(have_image planet6 image2)
	(have_image planet7 image2)
	(have_image phenomenon8 thermograph0)
))

)
