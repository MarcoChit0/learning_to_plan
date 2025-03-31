; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 443253

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child205 child370 - child
    bread114 bread130 bread458 - bread-portion
    content323 content114 content32 - content-portion
    tray33 tray152 tray356 - tray
    table223 table73 table186 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray152 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread130)
     (at_kitchen_bread bread458)
     (at_kitchen_content content323)
     (at_kitchen_content content114)
     (at_kitchen_content content32)
     (not_allergic_gluten child344)
     (not_allergic_gluten child370)
     (not_allergic_gluten child205)
     (waiting child344 table186)
     (waiting child205 table223)
     (waiting child370 table73)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child344)
     (served child205)
     (served child370)
    )
  )
)
