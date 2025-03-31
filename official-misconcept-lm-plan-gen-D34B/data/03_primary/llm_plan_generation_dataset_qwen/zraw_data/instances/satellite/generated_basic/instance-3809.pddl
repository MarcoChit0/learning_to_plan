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
	satellite2 - satellite
	instrument7 - instrument
	infrared0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	groundstation2 - direction
	star1 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star4)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image planet6 spectrograph2)
	(have_image phenomenon7 infrared0)
	(have_image star8 spectrograph2)
	(have_image planet9 infrared0)
))

)
