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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	image0 - mode
	infrared1 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 groundstation2)
	(have_image phenomenon3 infrared1)
	(have_image planet4 image0)
))

)
