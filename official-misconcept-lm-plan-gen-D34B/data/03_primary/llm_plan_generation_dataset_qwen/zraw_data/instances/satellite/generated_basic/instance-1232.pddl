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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	thermograph1 - mode
	infrared3 - mode
	thermograph0 - mode
	infrared2 - mode
	star3 - direction
	star4 - direction
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 infrared3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image planet5 infrared2)
	(have_image planet6 infrared2)
	(have_image star7 infrared3)
	(have_image phenomenon8 thermograph0)
))

)
