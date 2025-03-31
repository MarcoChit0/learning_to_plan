(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	spectrograph3 - mode
	infrared1 - mode
	image2 - mode
	infrared0 - mode
	star4 - direction
	star2 - direction
	star3 - direction
	star1 - direction
	star0 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon10)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(calibration_target instrument1 star3)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star1)
	(supports instrument5 image2)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 image2)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 image2)
	(calibration_target instrument10 star1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star1)
	(supports instrument12 spectrograph3)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star0)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 star8)
	(pointing satellite4 star0)
	(have_image planet5 image2)
	(have_image star6 image2)
	(have_image planet7 infrared0)
	(have_image star8 spectrograph3)
	(have_image planet9 infrared1)
	(have_image phenomenon10 image2)
))

)
