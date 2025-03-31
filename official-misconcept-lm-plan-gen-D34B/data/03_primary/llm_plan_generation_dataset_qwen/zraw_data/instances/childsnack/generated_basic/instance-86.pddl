; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 394595

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child487 child106 - child
    bread34 bread299 - bread-portion
    content257 content231 - content-portion
    tray94 tray83 - tray
    table432 table213 table428 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray94 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread299)
     (at_kitchen_content content257)
     (at_kitchen_content content231)
     (not_allergic_gluten child487)
     (not_allergic_gluten child106)
     (waiting child487 table428)
     (waiting child106 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child487)
     (served child106)
    )
  )
)
