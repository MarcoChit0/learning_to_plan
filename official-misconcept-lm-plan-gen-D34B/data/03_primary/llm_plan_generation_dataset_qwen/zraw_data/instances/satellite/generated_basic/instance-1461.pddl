(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image star4 image0)
	(have_image star5 image0)
	(have_image planet6 image0)
))

)
