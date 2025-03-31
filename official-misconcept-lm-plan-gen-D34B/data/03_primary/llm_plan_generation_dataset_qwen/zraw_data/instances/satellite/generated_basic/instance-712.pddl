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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph0 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon3)
)
(:goal (and
	(pointing satellite0 phenomenon3)
	(pointing satellite3 phenomenon3)
	(have_image star2 thermograph0)
	(have_image phenomenon3 infrared1)
))

)
