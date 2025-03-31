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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph0 - mode
	thermograph3 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation0 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument3 thermograph3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 thermograph3)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 thermograph2)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 thermograph3)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
)
(:goal (and
	(pointing satellite0 planet8)
	(pointing satellite3 groundstation2)
	(have_image phenomenon4 spectrograph0)
	(have_image phenomenon5 thermograph2)
	(have_image star6 thermograph2)
	(have_image star7 thermograph3)
	(have_image planet8 thermograph2)
	(have_image phenomenon9 spectrograph0)
))

)
