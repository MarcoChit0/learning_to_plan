; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 319104

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child499 - child
    bread99 bread363 - bread-portion
    content375 content76 - content-portion
    tray336 tray313 tray397 - tray
    table494 table59 table124 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at tray313 kitchen)
     (at tray397 kitchen)
     (at_kitchen_bread bread99)
     (at_kitchen_bread bread363)
     (at_kitchen_content content375)
     (at_kitchen_content content76)
     (not_allergic_gluten child246)
     (not_allergic_gluten child499)
     (waiting child246 table494)
     (waiting child499 table59)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child246)
     (served child499)
    )
  )
)
