; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 594810

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 child301 - child
    bread125 bread240 - bread-portion
    content307 content400 - content-portion
    tray471 tray448 tray117 tray130 - tray
    table403 table349 table21 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray448 kitchen)
     (at tray117 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread125)
     (at_kitchen_bread bread240)
     (at_kitchen_content content307)
     (at_kitchen_content content400)
     (not_allergic_gluten child301)
     (not_allergic_gluten child433)
     (waiting child433 table349)
     (waiting child301 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child433)
     (served child301)
    )
  )
)
