(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph1 - mode
	infrared0 - mode
	image2 - mode
	star2 - direction
	star4 - direction
	groundstation3 - direction
	groundstation0 - direction
	star1 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument4 thermograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument5 image2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument6 infrared0)
	(supports instrument6 image2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image2)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite2 planet8)
	(have_image phenomenon5 image2)
	(have_image phenomenon6 thermograph1)
	(have_image planet7 thermograph1)
	(have_image planet8 thermograph1)
))

)
