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
	infrared1 - mode
	image2 - mode
	infrared0 - mode
	groundstation3 - direction
	star0 - direction
	star1 - direction
	star2 - direction
	planet4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(calibration_target instrument0 star2)
	(supports instrument1 image2)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(supports instrument4 image2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image2)
	(calibration_target instrument7 star0)
	(supports instrument8 image2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument9 image2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(supports instrument10 image2)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star1)
	(supports instrument12 image2)
	(supports instrument12 infrared0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
)
(:goal (and
	(pointing satellite2 star2)
	(have_image planet4 image2)
	(have_image phenomenon5 image2)
	(have_image planet6 image2)
	(have_image phenomenon7 infrared1)
	(have_image planet8 infrared1)
))

)
