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
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	spectrograph1 - mode
	image0 - mode
	star2 - direction
	star1 - direction
	groundstation3 - direction
	groundstation6 - direction
	star0 - direction
	groundstation4 - direction
	groundstation5 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
	phenomenon10 - direction
	planet11 - direction
	star12 - direction
	phenomenon13 - direction
	star14 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation6)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation6)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument10 spectrograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation6)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation5)
	(calibration_target instrument12 groundstation4)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet9)
)
(:goal (and
	(pointing satellite2 groundstation6)
	(have_image star7 spectrograph1)
	(have_image phenomenon8 spectrograph1)
	(have_image planet9 spectrograph1)
	(have_image phenomenon10 image0)
	(have_image planet11 spectrograph1)
	(have_image star12 spectrograph1)
	(have_image phenomenon13 image0)
	(have_image star14 image0)
))

)
