(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite3 groundstation2)
	(have_image phenomenon4 image0)
	(have_image star5 image0)
	(have_image planet6 image0)
))

)
