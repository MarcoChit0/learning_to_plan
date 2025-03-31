(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	star5 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 thermograph2)
	(have_image planet8 thermograph2)
))

)
