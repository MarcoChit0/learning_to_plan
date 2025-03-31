; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 559514

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child252 - child
    bread492 bread282 - bread-portion
    content467 content423 - content-portion
    tray4 tray216 - tray
    table174 table170 table404 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray4 kitchen)
     (at tray216 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread282)
     (at_kitchen_content content467)
     (at_kitchen_content content423)
     (not_allergic_gluten child252)
     (not_allergic_gluten child200)
     (waiting child200 table170)
     (waiting child252 table174)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child200)
     (served child252)
    )
  )
)
