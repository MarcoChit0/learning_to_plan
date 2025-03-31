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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	star1 - direction
	groundstation3 - direction
	star4 - direction
	star0 - direction
	groundstation2 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
)
(:goal (and
	(pointing satellite2 planet7)
	(have_image phenomenon5 thermograph1)
	(have_image star6 thermograph1)
	(have_image planet7 thermograph1)
))

)
