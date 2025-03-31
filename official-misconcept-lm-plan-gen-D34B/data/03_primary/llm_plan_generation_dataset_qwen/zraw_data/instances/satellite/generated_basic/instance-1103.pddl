(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared0 - mode
	star1 - direction
	star0 - direction
	groundstation4 - direction
	star3 - direction
	groundstation2 - direction
	groundstation5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(calibration_target instrument8 groundstation5)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 star3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation2)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star3)
	(calibration_target instrument17 groundstation4)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation5)
	(calibration_target instrument18 groundstation2)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet6)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite2 groundstation4)
	(pointing satellite3 groundstation4)
	(pointing satellite5 star1)
	(have_image planet6 infrared0)
))

)
