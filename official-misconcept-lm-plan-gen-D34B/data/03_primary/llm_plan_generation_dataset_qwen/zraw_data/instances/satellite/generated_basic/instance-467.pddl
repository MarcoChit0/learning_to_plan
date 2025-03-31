(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image3 - mode
	image0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	star1 - direction
	groundstation4 - direction
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 image3)
	(supports instrument7 spectrograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
)
(:goal (and
	(pointing satellite0 planet5)
	(pointing satellite1 groundstation4)
	(have_image planet5 spectrograph2)
	(have_image star6 image3)
	(have_image phenomenon7 spectrograph1)
))

)
