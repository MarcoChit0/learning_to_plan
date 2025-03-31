; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 906147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 child70 - child
    bread96 bread88 - bread-portion
    content309 content231 - content-portion
    tray454 tray356 tray48 - tray
    table460 table31 table74 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray454 kitchen)
     (at tray356 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread88)
     (at_kitchen_content content309)
     (at_kitchen_content content231)
     (not_allergic_gluten child70)
     (not_allergic_gluten child380)
     (waiting child380 table74)
     (waiting child70 table31)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child380)
     (served child70)
    )
  )
)
