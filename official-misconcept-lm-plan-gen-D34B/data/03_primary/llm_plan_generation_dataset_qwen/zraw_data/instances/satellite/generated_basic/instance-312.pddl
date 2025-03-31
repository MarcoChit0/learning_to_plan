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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph3 - mode
	image0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation2 - direction
	groundstation3 - direction
	groundstation5 - direction
	star1 - direction
	groundstation0 - direction
	star4 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(have_image planet6 spectrograph2)
))

)
