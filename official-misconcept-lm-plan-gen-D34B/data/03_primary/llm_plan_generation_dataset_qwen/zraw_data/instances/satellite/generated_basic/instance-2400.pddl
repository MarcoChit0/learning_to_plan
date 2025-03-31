(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph3 - mode
	image1 - mode
	image0 - mode
	infrared2 - mode
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	planet5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument7 image1)
	(supports instrument7 infrared2)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph3)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite2 planet6)
	(have_image planet5 thermograph3)
	(have_image planet6 thermograph3)
))

)
