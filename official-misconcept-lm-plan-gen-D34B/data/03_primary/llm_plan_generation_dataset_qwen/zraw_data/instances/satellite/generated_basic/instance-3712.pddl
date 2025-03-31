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
	spectrograph0 - mode
	thermograph1 - mode
	spectrograph2 - mode
	star3 - direction
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite1 groundstation4)
	(have_image phenomenon6 spectrograph2)
))

)
