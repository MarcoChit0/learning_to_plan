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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	spectrograph0 - mode
	infrared1 - mode
	thermograph2 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(pointing satellite1 star4)
	(pointing satellite2 star0)
	(have_image phenomenon2 infrared1)
	(have_image star3 spectrograph0)
	(have_image star4 thermograph2)
))

)
