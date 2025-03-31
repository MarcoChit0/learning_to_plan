; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 274407

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child358 child270 - child
    bread113 bread173 - bread-portion
    content388 content467 - content-portion
    tray343 tray172 tray229 tray390 - tray
    table381 table290 table83 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray172 kitchen)
     (at tray229 kitchen)
     (at tray390 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread173)
     (at_kitchen_content content388)
     (at_kitchen_content content467)
     (not_allergic_gluten child358)
     (not_allergic_gluten child270)
     (waiting child358 table290)
     (waiting child270 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child358)
     (served child270)
    )
  )
)
