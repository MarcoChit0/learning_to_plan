(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared0 - mode
	star4 - direction
	groundstation5 - direction
	groundstation3 - direction
	star2 - direction
	star0 - direction
	star1 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation5)
	(calibration_target instrument8 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star4)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star4)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 groundstation5)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation5)
	(calibration_target instrument17 star4)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation5)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation3)
	(calibration_target instrument19 star2)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star2)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star1)
	(calibration_target instrument21 star0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation3)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite3 star1)
	(have_image planet6 infrared0)
))

)
