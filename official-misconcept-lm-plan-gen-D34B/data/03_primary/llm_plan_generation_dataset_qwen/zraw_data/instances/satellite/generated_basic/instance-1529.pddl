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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	image0 - mode
	star2 - direction
	groundstation5 - direction
	star3 - direction
	groundstation1 - direction
	star0 - direction
	star4 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet8)
	(supports instrument12 image0)
	(calibration_target instrument12 star4)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star2)
	(supports instrument14 image0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation5)
	(calibration_target instrument14 star3)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star3)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument16 infrared1)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star4)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite3 groundstation5)
	(have_image planet6 infrared1)
	(have_image star7 infrared1)
	(have_image planet8 infrared1)
))

)
