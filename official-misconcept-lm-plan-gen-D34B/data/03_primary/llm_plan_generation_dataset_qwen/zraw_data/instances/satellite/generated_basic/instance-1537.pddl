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
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	image2 - mode
	image0 - mode
	infrared1 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet1 image2)
))

)
