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
	instrument10 - instrument
	instrument11 - instrument
	infrared1 - mode
	image0 - mode
	star2 - direction
	groundstation0 - direction
	star1 - direction
	phenomenon3 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 phenomenon3)
	(have_image phenomenon3 infrared1)
	(have_image star4 image0)
	(have_image planet5 infrared1)
	(have_image planet6 infrared1)
	(have_image planet7 image0)
))

)
