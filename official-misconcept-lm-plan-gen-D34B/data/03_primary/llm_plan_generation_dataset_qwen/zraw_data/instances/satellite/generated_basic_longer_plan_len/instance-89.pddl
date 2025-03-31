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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	spectrograph0 - mode
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation5 - direction
	groundstation2 - direction
	star4 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 star4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation5)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star4)
	(calibration_target instrument12 groundstation2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite0 phenomenon10)
	(pointing satellite2 groundstation0)
	(pointing satellite3 planet9)
	(have_image star6 spectrograph0)
	(have_image star7 spectrograph0)
	(have_image planet8 spectrograph0)
	(have_image planet9 spectrograph0)
	(have_image phenomenon10 spectrograph0)
	(have_image planet11 spectrograph0)
))

)
