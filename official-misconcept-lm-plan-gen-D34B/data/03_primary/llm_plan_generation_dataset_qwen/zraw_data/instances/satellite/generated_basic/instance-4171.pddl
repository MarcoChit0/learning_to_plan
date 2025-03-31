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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image0 - mode
	image1 - mode
	groundstation2 - direction
	star4 - direction
	groundstation1 - direction
	star3 - direction
	groundstation5 - direction
	groundstation0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star4)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 star4)
	(supports instrument11 image1)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 image1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite2 groundstation5)
	(have_image phenomenon6 image0)
))

)
