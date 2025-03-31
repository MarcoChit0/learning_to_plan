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
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	infrared2 - mode
	thermograph1 - mode
	infrared0 - mode
	infrared3 - mode
	groundstation3 - direction
	groundstation4 - direction
	groundstation0 - direction
	star1 - direction
	star2 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared0)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument4 infrared2)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared2)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 infrared2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared2)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite1 star2)
	(pointing satellite2 star1)
	(pointing satellite3 groundstation0)
	(have_image planet5 thermograph1)
	(have_image star6 thermograph1)
	(have_image phenomenon7 infrared3)
))

)
