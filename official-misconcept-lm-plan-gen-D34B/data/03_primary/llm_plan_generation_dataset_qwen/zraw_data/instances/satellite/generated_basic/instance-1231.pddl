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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	thermograph0 - mode
	image2 - mode
	image1 - mode
	infrared3 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared3)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 image2)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared3)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image1)
	(supports instrument11 image2)
	(calibration_target instrument11 star4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star4)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star0)
	(supports instrument14 image2)
	(supports instrument14 thermograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 star4)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(have_image planet5 image1)
	(have_image planet6 image1)
	(have_image phenomenon7 image2)
	(have_image star8 thermograph0)
))

)
