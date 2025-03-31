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
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	infrared0 - mode
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation4)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star2)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star2)
	(supports instrument21 infrared0)
	(calibration_target instrument21 groundstation1)
	(supports instrument22 infrared0)
	(calibration_target instrument22 groundstation1)
	(supports instrument23 infrared0)
	(calibration_target instrument23 groundstation0)
	(supports instrument24 infrared0)
	(calibration_target instrument24 groundstation1)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet6)
)
(:goal (and
	(pointing satellite0 star5)
	(pointing satellite3 groundstation0)
	(pointing satellite5 groundstation3)
	(have_image star5 infrared0)
	(have_image planet6 infrared0)
	(have_image planet7 infrared0)
))

)
