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
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	infrared0 - mode
	groundstation4 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 star3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star3)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(calibration_target instrument15 star2)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star3)
	(calibration_target instrument16 star1)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation5)
	(calibration_target instrument17 groundstation0)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation5)
)
(:goal (and
	(pointing satellite4 groundstation0)
	(pointing satellite5 groundstation4)
	(have_image planet6 infrared0)
))

)
