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
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	instrument21 - instrument
	infrared0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star1)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet2)
)
(:goal (and
	(have_image planet2 infrared0)
	(have_image star3 infrared0)
	(have_image phenomenon4 infrared0)
))

)
