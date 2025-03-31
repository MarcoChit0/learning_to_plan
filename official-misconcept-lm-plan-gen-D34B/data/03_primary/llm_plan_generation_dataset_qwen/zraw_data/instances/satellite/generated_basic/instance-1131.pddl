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
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	infrared0 - mode
	star3 - direction
	groundstation2 - direction
	star0 - direction
	star5 - direction
	groundstation4 - direction
	star1 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 star5)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 groundstation4)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(calibration_target instrument11 star5)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star5)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation4)
	(calibration_target instrument14 star1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star5)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation4)
	(calibration_target instrument17 star5)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite2 star1)
	(pointing satellite4 star1)
	(have_image star6 infrared0)
))

)
