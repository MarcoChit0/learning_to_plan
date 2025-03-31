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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image2 - mode
	infrared0 - mode
	infrared3 - mode
	thermograph1 - mode
	star0 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image2)
	(supports instrument1 infrared0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument3 infrared0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument6 image2)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared3)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 infrared0)
	(supports instrument11 image2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image2)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite1 planet7)
	(pointing satellite2 groundstation3)
	(have_image planet5 infrared3)
	(have_image planet6 image2)
	(have_image planet7 image2)
	(have_image star8 thermograph1)
))

)
