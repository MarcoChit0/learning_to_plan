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
	instrument7 - instrument
	spectrograph3 - mode
	spectrograph1 - mode
	infrared0 - mode
	infrared2 - mode
	star1 - direction
	star2 - direction
	groundstation3 - direction
	groundstation4 - direction
	star0 - direction
	planet5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	star9 - direction
	star10 - direction
	star11 - direction
	planet12 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star11)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star9)
)
(:goal (and
	(have_image planet5 infrared2)
	(have_image phenomenon6 infrared2)
	(have_image star7 spectrograph1)
	(have_image star8 infrared2)
	(have_image star9 spectrograph3)
	(have_image star10 spectrograph1)
	(have_image star11 infrared2)
	(have_image planet12 spectrograph3)
))

)
