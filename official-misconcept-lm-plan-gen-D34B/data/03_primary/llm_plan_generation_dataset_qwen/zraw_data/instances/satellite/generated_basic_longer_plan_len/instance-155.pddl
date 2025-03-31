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
	instrument7 - instrument
	instrument8 - instrument
	image1 - mode
	infrared2 - mode
	spectrograph3 - mode
	thermograph0 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
	star10 - direction
	star11 - direction
	phenomenon12 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph3)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph3)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star11)
	(supports instrument3 spectrograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 thermograph0)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star5 infrared2)
	(have_image planet6 thermograph0)
	(have_image star7 image1)
	(have_image planet8 spectrograph3)
	(have_image planet9 image1)
	(have_image star10 thermograph0)
	(have_image star11 spectrograph3)
	(have_image phenomenon12 spectrograph3)
))

)
