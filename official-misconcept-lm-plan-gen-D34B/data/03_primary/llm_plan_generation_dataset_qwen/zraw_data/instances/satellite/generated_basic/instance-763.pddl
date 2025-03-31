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
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star3 - direction
	groundstation4 - direction
	groundstation5 - direction
	star2 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation5)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image0)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite3 groundstation5)
	(have_image planet6 spectrograph1)
	(have_image star7 spectrograph2)
	(have_image planet8 spectrograph1)
	(have_image phenomenon9 image0)
))

)
