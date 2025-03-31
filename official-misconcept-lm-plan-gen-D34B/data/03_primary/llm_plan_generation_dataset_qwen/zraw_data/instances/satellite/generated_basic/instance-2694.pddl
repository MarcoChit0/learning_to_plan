(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	spectrograph3 - mode
	thermograph0 - mode
	infrared1 - mode
	thermograph2 - mode
	star3 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image phenomenon4 infrared1)
))

)
