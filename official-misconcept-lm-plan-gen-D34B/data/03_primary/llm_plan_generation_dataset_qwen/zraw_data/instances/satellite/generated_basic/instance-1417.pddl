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
	image2 - mode
	thermograph0 - mode
	infrared3 - mode
	thermograph1 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared3)
	(supports instrument6 image2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
)
(:goal (and
	(pointing satellite0 phenomenon1)
	(have_image phenomenon1 image2)
	(have_image planet2 infrared3)
	(have_image phenomenon3 image2)
))

)
