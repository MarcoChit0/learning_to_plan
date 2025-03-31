(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
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
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	infrared0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star1)
	(supports instrument22 infrared0)
	(calibration_target instrument22 groundstation0)
	(supports instrument23 infrared0)
	(calibration_target instrument23 groundstation0)
	(supports instrument24 infrared0)
	(calibration_target instrument24 star1)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite2 planet2)
	(have_image planet2 infrared0)
	(have_image planet3 infrared0)
	(have_image phenomenon4 infrared0)
))

)
