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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared3 - mode
	thermograph0 - mode
	spectrograph1 - mode
	spectrograph2 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	star5 - direction
	groundstation0 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite2 groundstation0)
	(have_image star6 spectrograph2)
))

)
