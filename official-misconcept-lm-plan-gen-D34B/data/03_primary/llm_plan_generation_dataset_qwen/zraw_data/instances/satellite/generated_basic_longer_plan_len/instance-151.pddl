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
	infrared1 - mode
	image3 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation2 - direction
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	star1 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
	planet10 - direction
	phenomenon11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph2)
	(supports instrument2 image3)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph2)
	(supports instrument6 image3)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
)
(:goal (and
	(have_image phenomenon5 infrared1)
	(have_image planet6 infrared1)
	(have_image planet7 spectrograph0)
	(have_image star8 spectrograph0)
	(have_image planet9 spectrograph2)
	(have_image planet10 spectrograph2)
	(have_image phenomenon11 infrared1)
	(have_image phenomenon12 image3)
))

)
