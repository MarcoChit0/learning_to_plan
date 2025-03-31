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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	spectrograph0 - mode
	groundstation5 - direction
	star6 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
	star11 - direction
	planet12 - direction
	phenomenon13 - direction
	planet14 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star6)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet12)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument6 image1)
	(calibration_target instrument6 star4)
	(supports instrument7 image1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 star4)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite2 groundstation0)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 image1)
	(have_image phenomenon9 image1)
	(have_image phenomenon10 spectrograph0)
	(have_image star11 spectrograph0)
	(have_image planet12 image1)
	(have_image phenomenon13 image1)
	(have_image planet14 image1)
))

)
