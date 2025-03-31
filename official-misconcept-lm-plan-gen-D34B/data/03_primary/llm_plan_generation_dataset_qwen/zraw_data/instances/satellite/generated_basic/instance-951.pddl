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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	infrared2 - mode
	image1 - mode
	image0 - mode
	thermograph3 - mode
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 infrared2)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
)
(:goal (and
	(have_image phenomenon4 thermograph3)
	(have_image phenomenon5 infrared2)
	(have_image star6 thermograph3)
))

)
