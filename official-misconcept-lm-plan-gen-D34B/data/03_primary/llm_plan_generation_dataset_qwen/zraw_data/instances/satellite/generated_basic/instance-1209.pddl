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
	instrument5 - instrument
	instrument6 - instrument
	thermograph1 - mode
	infrared0 - mode
	spectrograph2 - mode
	image3 - mode
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(supports instrument4 image3)
	(calibration_target instrument4 star2)
	(supports instrument5 image3)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image3)
	(supports instrument6 infrared0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 groundstation0)
	(have_image star3 image3)
	(have_image planet4 image3)
	(have_image phenomenon5 thermograph1)
))

)
