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
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared1 - mode
	infrared2 - mode
	image0 - mode
	star5 - direction
	star3 - direction
	groundstation2 - direction
	groundstation4 - direction
	star0 - direction
	star1 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 star5)
	(supports instrument1 image0)
	(supports instrument1 infrared2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared2)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 infrared2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(calibration_target instrument8 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument9 infrared2)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 star5)
	(supports instrument11 infrared2)
	(supports instrument11 image0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star5)
	(supports instrument12 infrared2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation2)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared1)
	(supports instrument13 infrared2)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite2 groundstation4)
	(have_image star6 image0)
))

)
