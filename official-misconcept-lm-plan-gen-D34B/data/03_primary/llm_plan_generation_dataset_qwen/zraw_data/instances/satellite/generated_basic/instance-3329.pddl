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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	image1 - mode
	infrared0 - mode
	infrared2 - mode
	star2 - direction
	groundstation0 - direction
	star1 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(supports instrument7 infrared2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared0)
	(supports instrument9 infrared2)
	(supports instrument9 image1)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared2)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared0)
	(supports instrument11 image1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 star1)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite2 star3)
	(have_image star3 image1)
	(have_image phenomenon4 infrared0)
))

)
