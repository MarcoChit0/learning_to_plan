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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph2 - mode
	thermograph1 - mode
	thermograph0 - mode
	groundstation4 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star5)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
)
(:goal (and
	(have_image planet6 spectrograph2)
	(have_image planet7 spectrograph2)
	(have_image planet8 spectrograph2)
	(have_image phenomenon9 thermograph1)
))

)
