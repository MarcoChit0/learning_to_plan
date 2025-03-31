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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	thermograph3 - mode
	spectrograph0 - mode
	image2 - mode
	spectrograph1 - mode
	groundstation3 - direction
	star0 - direction
	star4 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image2)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation2)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(have_image phenomenon5 image2)
	(have_image planet6 spectrograph0)
))

)
