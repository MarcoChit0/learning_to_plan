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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	image0 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation0 - direction
	star4 - direction
	star2 - direction
	groundstation3 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 star4)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite2 star4)
	(have_image phenomenon5 image0)
	(have_image phenomenon6 infrared1)
	(have_image star7 infrared1)
))

)
