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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	thermograph0 - mode
	infrared2 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite1 phenomenon3)
	(pointing satellite2 groundstation1)
	(have_image phenomenon2 infrared2)
	(have_image phenomenon3 infrared2)
	(have_image star4 thermograph0)
))

)
