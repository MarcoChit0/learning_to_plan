(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	star1 - direction
	groundstation5 - direction
	groundstation2 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 star4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation2)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 infrared1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation3)
	(calibration_target instrument13 star4)
	(supports instrument14 infrared1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation2)
	(calibration_target instrument15 star1)
	(supports instrument16 infrared1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation2)
	(calibration_target instrument16 groundstation5)
	(supports instrument17 infrared1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation2)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite1 star4)
	(pointing satellite2 star4)
	(have_image star6 infrared1)
	(have_image star7 infrared0)
	(have_image planet8 infrared1)
	(have_image star9 infrared1)
))

)
