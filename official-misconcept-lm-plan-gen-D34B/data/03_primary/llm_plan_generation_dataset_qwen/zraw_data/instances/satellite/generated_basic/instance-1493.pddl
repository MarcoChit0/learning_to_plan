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
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared0 - mode
	infrared1 - mode
	groundstation5 - direction
	groundstation0 - direction
	star3 - direction
	groundstation1 - direction
	star4 - direction
	star2 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 groundstation0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation5)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument11 infrared1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(calibration_target instrument12 star4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star4)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 star3)
	(supports instrument16 infrared0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star4)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star2)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation5)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite3 phenomenon7)
	(pointing satellite5 groundstation0)
	(have_image planet6 infrared1)
	(have_image phenomenon7 infrared1)
	(have_image star8 infrared0)
))

)
