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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	image2 - mode
	image0 - mode
	infrared1 - mode
	star1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 image2)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image2)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image2)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument9 image0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image phenomenon2 image0)
))

)
