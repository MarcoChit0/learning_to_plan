; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 46310

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child450 - child
    bread443 bread305 - bread-portion
    content420 content422 - content-portion
    tray185 tray363 tray451 - tray
    table476 table178 table253 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray363 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread305)
     (at_kitchen_content content420)
     (at_kitchen_content content422)
     (not_allergic_gluten child450)
     (not_allergic_gluten child56)
     (waiting child56 table253)
     (waiting child450 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child56)
     (served child450)
    )
  )
)
