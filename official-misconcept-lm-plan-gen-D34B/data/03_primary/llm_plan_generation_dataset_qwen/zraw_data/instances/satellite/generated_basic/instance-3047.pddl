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
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared0 - mode
	infrared1 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument14 infrared1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 infrared0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 star1)
	(supports instrument17 infrared0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star0)
	(supports instrument19 infrared0)
	(supports instrument19 infrared1)
	(calibration_target instrument19 groundstation2)
	(supports instrument20 infrared0)
	(supports instrument20 infrared1)
	(calibration_target instrument20 groundstation2)
	(supports instrument21 infrared0)
	(supports instrument21 infrared1)
	(calibration_target instrument21 groundstation3)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(have_image star4 infrared0)
	(have_image star5 infrared0)
	(have_image star6 infrared0)
))

)
