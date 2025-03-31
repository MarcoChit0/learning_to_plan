; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 668464

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child477 - child
    bread43 bread335 - bread-portion
    content484 content117 - content-portion
    tray455 tray332 tray478 - tray
    table359 table457 table99 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray455 kitchen)
     (at tray332 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread335)
     (at_kitchen_content content484)
     (at_kitchen_content content117)
     (not_allergic_gluten child477)
     (not_allergic_gluten child403)
     (waiting child403 table457)
     (waiting child477 table99)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child403)
     (served child477)
    )
  )
)
