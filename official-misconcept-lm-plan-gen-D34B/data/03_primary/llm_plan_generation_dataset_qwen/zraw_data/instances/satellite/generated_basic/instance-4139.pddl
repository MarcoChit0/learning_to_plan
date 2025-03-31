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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	image0 - mode
	infrared1 - mode
	groundstation2 - direction
	star3 - direction
	star5 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation4 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 star5)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 star5)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(have_image planet6 image0)
))

)
