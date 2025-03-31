(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared0 - mode
	thermograph2 - mode
	image1 - mode
	star2 - direction
	star0 - direction
	star4 - direction
	star1 - direction
	groundstation3 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite2 planet7)
	(have_image star6 image1)
	(have_image planet7 image1)
	(have_image phenomenon8 infrared0)
	(have_image phenomenon9 thermograph2)
))

)
