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
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	thermograph0 - mode
	thermograph3 - mode
	thermograph2 - mode
	thermograph1 - mode
	groundstation3 - direction
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph3)
	(supports instrument10 thermograph1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star9)
	(supports instrument12 thermograph2)
	(supports instrument12 thermograph3)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph2)
	(supports instrument13 thermograph3)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph1)
	(supports instrument15 thermograph2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image star4 thermograph0)
	(have_image phenomenon5 thermograph0)
	(have_image phenomenon6 thermograph1)
	(have_image phenomenon7 thermograph1)
	(have_image planet8 thermograph2)
	(have_image star9 thermograph3)
))

)
