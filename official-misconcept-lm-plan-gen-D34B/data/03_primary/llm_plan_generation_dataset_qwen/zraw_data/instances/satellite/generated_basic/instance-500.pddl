(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared0 - mode
	thermograph2 - mode
	image1 - mode
	image3 - mode
	star4 - direction
	groundstation3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph2)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 image3)
	(calibration_target instrument3 star2)
	(supports instrument4 image3)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument5 image1)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
	(supports instrument6 image1)
	(supports instrument6 image3)
	(calibration_target instrument6 star2)
	(supports instrument7 image3)
	(calibration_target instrument7 star0)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star2)
	(pointing satellite2 groundstation1)
	(have_image planet5 thermograph2)
	(have_image phenomenon6 thermograph2)
	(have_image phenomenon7 infrared0)
))

)
