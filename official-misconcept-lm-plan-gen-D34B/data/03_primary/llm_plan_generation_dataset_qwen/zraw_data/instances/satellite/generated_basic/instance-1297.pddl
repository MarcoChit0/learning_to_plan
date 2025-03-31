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
	satellite2 - satellite
	instrument6 - instrument
	thermograph1 - mode
	thermograph2 - mode
	spectrograph0 - mode
	groundstation1 - direction
	star2 - direction
	groundstation4 - direction
	star0 - direction
	star5 - direction
	star3 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star5)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 star5)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
)
(:goal (and
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 thermograph1)
	(have_image planet8 thermograph2)
	(have_image planet9 spectrograph0)
))

)
