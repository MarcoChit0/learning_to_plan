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
	instrument10 - instrument
	infrared1 - mode
	infrared2 - mode
	spectrograph0 - mode
	groundstation2 - direction
	groundstation1 - direction
	star4 - direction
	groundstation5 - direction
	star3 - direction
	groundstation0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument8 infrared1)
	(supports instrument8 infrared2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star3)
	(calibration_target instrument8 star4)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation0)
	(have_image phenomenon6 infrared2)
))

)
