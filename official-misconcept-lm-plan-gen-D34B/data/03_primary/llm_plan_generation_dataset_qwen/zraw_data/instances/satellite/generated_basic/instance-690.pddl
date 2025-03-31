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
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared0 - mode
	thermograph1 - mode
	image2 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 image2)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 planet3)
	(have_image phenomenon2 image2)
	(have_image planet3 image2)
))

)
