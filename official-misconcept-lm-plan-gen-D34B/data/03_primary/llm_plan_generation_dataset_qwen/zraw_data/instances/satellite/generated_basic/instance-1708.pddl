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
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	infrared0 - mode
	groundstation2 - direction
	star0 - direction
	star3 - direction
	star1 - direction
	groundstation4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(calibration_target instrument6 star3)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 infrared0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image phenomenon5 infrared0)
))

)
