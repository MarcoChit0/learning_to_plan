(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	thermograph1 - mode
	thermograph3 - mode
	image2 - mode
	infrared0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph3)
	(supports instrument2 thermograph1)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 infrared0)
	(supports instrument5 image2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph3)
	(supports instrument6 image2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(have_image star2 image2)
))

)
