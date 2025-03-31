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
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 infrared0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
	(supports instrument17 infrared0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 infrared1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 infrared1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation1)
	(supports instrument20 infrared1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 groundstation1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 groundstation1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet3)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite2 groundstation0)
	(have_image star2 infrared1)
	(have_image planet3 infrared1)
	(have_image star4 infrared0)
))

)
