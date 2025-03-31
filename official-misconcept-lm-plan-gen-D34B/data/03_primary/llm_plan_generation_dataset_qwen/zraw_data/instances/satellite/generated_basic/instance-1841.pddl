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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	infrared1 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 phenomenon7)
	(have_image star3 image0)
	(have_image phenomenon4 infrared1)
	(have_image phenomenon5 infrared1)
	(have_image planet6 image0)
	(have_image phenomenon7 image0)
))

)
