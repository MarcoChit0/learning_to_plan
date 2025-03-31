(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	infrared1 - mode
	infrared2 - mode
	thermograph3 - mode
	groundstation3 - direction
	groundstation1 - direction
	star4 - direction
	star2 - direction
	groundstation0 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 thermograph3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared2)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument6 image0)
	(supports instrument6 thermograph3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared2)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument8 image0)
	(supports instrument8 infrared2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared1)
	(supports instrument9 thermograph3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image star5 image0)
	(have_image planet6 infrared1)
))

)
