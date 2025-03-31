(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image3 - mode
	image2 - mode
	spectrograph1 - mode
	thermograph0 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph0)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 star4)
	(have_image star3 image3)
	(have_image star4 spectrograph1)
	(have_image phenomenon5 image2)
))

)
