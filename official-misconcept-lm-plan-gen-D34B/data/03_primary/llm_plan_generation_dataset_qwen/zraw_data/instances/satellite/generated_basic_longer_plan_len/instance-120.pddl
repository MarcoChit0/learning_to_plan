(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	image0 - mode
	star6 - direction
	star0 - direction
	groundstation1 - direction
	star5 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
	star12 - direction
	star13 - direction
	planet14 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star6)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star6)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 star5)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 image0)
	(supports instrument9 image1)
	(calibration_target instrument9 star4)
	(calibration_target instrument9 star5)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 groundstation2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
)
(:goal (and
	(pointing satellite2 planet14)
	(have_image planet7 image0)
	(have_image star8 image0)
	(have_image planet9 image1)
	(have_image phenomenon10 image1)
	(have_image planet11 image1)
	(have_image star12 image1)
	(have_image star13 image0)
	(have_image planet14 image1)
))

)
