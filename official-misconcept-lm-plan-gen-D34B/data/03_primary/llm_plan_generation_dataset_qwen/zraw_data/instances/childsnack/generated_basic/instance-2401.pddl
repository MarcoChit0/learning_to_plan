; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 660713

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child240 - child
    bread198 bread117 - bread-portion
    content102 content201 - content-portion
    tray53 - tray
    table391 table358 table270 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at_kitchen_bread bread198)
     (at_kitchen_bread bread117)
     (at_kitchen_content content102)
     (at_kitchen_content content201)
     (not_allergic_gluten child252)
     (not_allergic_gluten child240)
     (waiting child252 table270)
     (waiting child240 table270)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child252)
     (served child240)
    )
  )
)
