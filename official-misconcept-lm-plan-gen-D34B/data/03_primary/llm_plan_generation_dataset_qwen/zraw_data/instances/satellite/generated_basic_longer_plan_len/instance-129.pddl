(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite1 - satellite
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	spectrograph0 - mode
	star3 - direction
	star5 - direction
	star0 - direction
	groundstation1 - direction
	groundstation4 - direction
	star2 - direction
	groundstation6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
	star11 - direction
	star12 - direction
	star13 - direction
	star14 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation6)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation6)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star5)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation6)
	(calibration_target instrument6 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation4)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star14)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation6)
	(calibration_target instrument10 star2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite0 planet10)
	(pointing satellite2 planet8)
	(have_image star7 image1)
	(have_image planet8 spectrograph0)
	(have_image phenomenon9 spectrograph0)
	(have_image planet10 image1)
	(have_image star11 image1)
	(have_image star12 spectrograph0)
	(have_image star13 image1)
	(have_image star14 image1)
))

)
