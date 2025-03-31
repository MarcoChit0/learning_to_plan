(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	groundstation6 - direction
	star3 - direction
	star5 - direction
	groundstation2 - direction
	star0 - direction
	star4 - direction
	groundstation1 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	star10 - direction
	planet11 - direction
	star12 - direction
	planet13 - direction
	planet14 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation6)
	(calibration_target instrument2 star4)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet13)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation6)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation6)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star5)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 groundstation2)
	(have_image star7 spectrograph1)
	(have_image phenomenon8 spectrograph0)
	(have_image phenomenon9 spectrograph0)
	(have_image star10 spectrograph1)
	(have_image planet11 spectrograph0)
	(have_image star12 spectrograph0)
	(have_image planet13 spectrograph1)
	(have_image planet14 spectrograph1)
))

)
