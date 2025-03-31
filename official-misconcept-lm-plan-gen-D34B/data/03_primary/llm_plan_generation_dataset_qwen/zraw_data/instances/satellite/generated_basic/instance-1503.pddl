(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	infrared0 - mode
	groundstation4 - direction
	star0 - direction
	groundstation5 - direction
	star1 - direction
	star3 - direction
	groundstation2 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 groundstation5)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet8)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(calibration_target instrument13 star3)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet8)
	(supports instrument14 infrared0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation2)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star1)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 infrared1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star3)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 groundstation2)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet7)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite2 groundstation5)
	(have_image star6 infrared1)
	(have_image planet7 infrared1)
	(have_image planet8 infrared1)
))

)
