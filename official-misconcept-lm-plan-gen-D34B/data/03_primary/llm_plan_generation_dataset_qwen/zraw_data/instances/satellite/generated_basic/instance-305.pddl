(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	thermograph1 - mode
	image3 - mode
	thermograph2 - mode
	spectrograph0 - mode
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(supports instrument1 image3)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 image3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star5)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 image3)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image3)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 star5)
	(pointing satellite2 groundstation1)
	(have_image star6 thermograph1)
))

)
