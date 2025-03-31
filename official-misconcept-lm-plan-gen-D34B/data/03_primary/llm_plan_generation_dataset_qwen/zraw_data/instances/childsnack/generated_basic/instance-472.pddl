; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 926327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child20 - child
    bread283 bread79 - bread-portion
    content128 content311 - content-portion
    tray357 tray255 - tray
    table65 table475 table119 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray255 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_bread bread79)
     (at_kitchen_content content128)
     (at_kitchen_content content311)
     (not_allergic_gluten child20)
     (not_allergic_gluten child238)
     (waiting child238 table119)
     (waiting child20 table475)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child20)
    )
  )
)
