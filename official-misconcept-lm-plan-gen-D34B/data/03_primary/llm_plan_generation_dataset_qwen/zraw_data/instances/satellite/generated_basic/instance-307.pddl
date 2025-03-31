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
	thermograph2 - mode
	infrared3 - mode
	spectrograph1 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation3 - direction
	star5 - direction
	star4 - direction
	star2 - direction
	groundstation1 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument5 infrared3)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation1)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 star0)
	(have_image phenomenon6 spectrograph1)
))

)
