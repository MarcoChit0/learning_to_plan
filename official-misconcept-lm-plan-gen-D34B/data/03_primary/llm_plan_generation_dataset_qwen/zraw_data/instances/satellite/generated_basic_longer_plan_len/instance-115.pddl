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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	thermograph1 - mode
	spectrograph0 - mode
	star1 - direction
	star3 - direction
	groundstation6 - direction
	star4 - direction
	star0 - direction
	star2 - direction
	groundstation5 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
	planet12 - direction
	star13 - direction
	phenomenon14 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation6)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation6)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet11)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation6)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star4)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation5)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
)
(:goal (and
	(have_image star7 spectrograph0)
	(have_image planet8 thermograph1)
	(have_image planet9 thermograph1)
	(have_image phenomenon10 spectrograph0)
	(have_image planet11 thermograph1)
	(have_image planet12 spectrograph0)
	(have_image star13 spectrograph0)
	(have_image phenomenon14 spectrograph0)
))

)
