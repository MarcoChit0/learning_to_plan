; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 209467

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child97 - child
    bread336 bread230 - bread-portion
    content347 content354 - content-portion
    tray278 tray309 tray130 - tray
    table449 table447 table399 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray278 kitchen)
     (at tray309 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread336)
     (at_kitchen_bread bread230)
     (at_kitchen_content content347)
     (at_kitchen_content content354)
     (no_gluten_bread bread336)
     (no_gluten_content content354)
     (allergic_gluten child442)
     (not_allergic_gluten child97)
     (waiting child442 table449)
     (waiting child97 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child97)
    )
  )
)
