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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	image0 - mode
	infrared1 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star0)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
)
(:goal (and
	(have_image phenomenon2 infrared1)
	(have_image planet3 image0)
))

)
