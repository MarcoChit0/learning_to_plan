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
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	thermograph1 - mode
	infrared0 - mode
	infrared2 - mode
	star2 - direction
	star4 - direction
	groundstation5 - direction
	star1 - direction
	star0 - direction
	groundstation3 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
	star10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet11)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star4)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation5)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument9 infrared0)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 star1)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
)
(:goal (and
	(pointing satellite1 groundstation5)
	(have_image star6 infrared0)
	(have_image planet7 infrared2)
	(have_image star8 infrared2)
	(have_image star9 infrared0)
	(have_image star10 infrared0)
	(have_image planet11 infrared2)
))

)
