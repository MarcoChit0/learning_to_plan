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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	thermograph2 - mode
	infrared1 - mode
	thermograph3 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	groundstation3 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite2 phenomenon7)
	(have_image phenomenon4 infrared1)
	(have_image planet5 thermograph3)
	(have_image star6 spectrograph0)
	(have_image phenomenon7 thermograph2)
	(have_image phenomenon8 spectrograph0)
	(have_image phenomenon9 thermograph2)
))

)
