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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	infrared0 - mode
	image2 - mode
	spectrograph1 - mode
	star3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation2 - direction
	star5 - direction
	star4 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star5)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star5)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 spectrograph1)
	(supports instrument5 image2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 image2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument8 image2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 star5)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image phenomenon6 image2)
	(have_image phenomenon7 spectrograph1)
	(have_image phenomenon8 image2)
	(have_image phenomenon9 image2)
))

)
