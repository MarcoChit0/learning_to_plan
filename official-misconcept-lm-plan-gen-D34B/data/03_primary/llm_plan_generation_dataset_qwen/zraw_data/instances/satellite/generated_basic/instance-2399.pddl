(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared0 - mode
	image1 - mode
	image3 - mode
	image2 - mode
	groundstation3 - direction
	star1 - direction
	groundstation4 - direction
	star2 - direction
	star0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 image3)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 infrared0)
	(supports instrument7 image3)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image3)
	(supports instrument9 image2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(supports instrument10 image3)
	(supports instrument10 infrared0)
	(supports instrument10 image2)
	(calibration_target instrument10 star2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument11 image2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared0)
	(supports instrument12 image3)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(supports instrument13 image3)
	(supports instrument13 infrared0)
	(supports instrument13 image1)
	(calibration_target instrument13 star2)
	(supports instrument14 image1)
	(calibration_target instrument14 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite2 star2)
	(have_image phenomenon5 image3)
	(have_image phenomenon6 infrared0)
))

)
