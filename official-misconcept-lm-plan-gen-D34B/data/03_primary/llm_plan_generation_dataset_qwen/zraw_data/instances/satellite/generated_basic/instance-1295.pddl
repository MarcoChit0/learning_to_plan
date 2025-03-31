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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation4 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star5 - direction
	star3 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph2)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image planet6 thermograph1)
	(have_image star7 thermograph1)
	(have_image star8 thermograph1)
	(have_image planet9 image0)
))

)
