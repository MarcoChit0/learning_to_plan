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
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite5 - satellite
	instrument18 - instrument
	infrared0 - mode
	groundstation2 - direction
	groundstation4 - direction
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation0)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet7)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 groundstation2)
	(have_image planet5 infrared0)
	(have_image planet6 infrared0)
	(have_image planet7 infrared0)
	(have_image planet8 infrared0)
	(have_image planet9 infrared0)
	(have_image star10 infrared0)
))

)
