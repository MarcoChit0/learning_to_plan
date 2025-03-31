(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	infrared1 - mode
	infrared0 - mode
	star2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument11 infrared1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument13 infrared0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite3 phenomenon5)
	(have_image star4 infrared0)
	(have_image phenomenon5 infrared1)
	(have_image phenomenon6 infrared0)
))

)
