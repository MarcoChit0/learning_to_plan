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
	infrared0 - mode
	image3 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(supports instrument4 image3)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared0)
	(supports instrument5 image3)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon3 infrared0)
	(have_image phenomenon4 thermograph1)
	(have_image phenomenon5 thermograph1)
))

)
