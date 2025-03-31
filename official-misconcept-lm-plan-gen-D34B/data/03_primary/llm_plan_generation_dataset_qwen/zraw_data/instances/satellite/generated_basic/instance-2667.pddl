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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	spectrograph0 - mode
	groundstation4 - direction
	star1 - direction
	groundstation0 - direction
	star5 - direction
	star3 - direction
	groundstation2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star5)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 star5)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite3 groundstation4)
	(have_image planet6 spectrograph0)
))

)
