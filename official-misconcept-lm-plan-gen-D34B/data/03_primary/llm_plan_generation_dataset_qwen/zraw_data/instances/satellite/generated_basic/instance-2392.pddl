(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph1 - mode
	infrared3 - mode
	infrared2 - mode
	thermograph0 - mode
	groundstation4 - direction
	groundstation0 - direction
	star2 - direction
	star1 - direction
	groundstation3 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared2)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 planet6)
	(pointing satellite2 star1)
	(have_image star5 thermograph0)
	(have_image planet6 infrared2)
))

)
