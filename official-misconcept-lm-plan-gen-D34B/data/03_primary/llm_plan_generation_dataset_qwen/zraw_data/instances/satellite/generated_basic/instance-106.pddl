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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	star4 - direction
	groundstation3 - direction
	groundstation5 - direction
	star2 - direction
	star1 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star4)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star4)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument13 infrared1)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(calibration_target instrument13 star1)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star1)
	(supports instrument15 infrared1)
	(calibration_target instrument15 star2)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 image0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 star1)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 groundstation5)
	(pointing satellite3 phenomenon9)
	(have_image star6 infrared1)
	(have_image phenomenon7 image0)
	(have_image phenomenon8 infrared1)
	(have_image phenomenon9 infrared1)
))

)
