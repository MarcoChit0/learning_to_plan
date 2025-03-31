; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 87967

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 child295 - child
    bread427 bread202 - bread-portion
    content376 content164 - content-portion
    tray369 tray121 - tray
    table415 table34 table343 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray121 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_bread bread202)
     (at_kitchen_content content376)
     (at_kitchen_content content164)
     (not_allergic_gluten child295)
     (not_allergic_gluten child160)
     (waiting child160 table343)
     (waiting child295 table343)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child160)
     (served child295)
    )
  )
)
