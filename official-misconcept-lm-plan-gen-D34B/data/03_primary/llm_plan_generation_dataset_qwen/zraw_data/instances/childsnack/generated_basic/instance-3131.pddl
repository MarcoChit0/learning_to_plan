; child-snack task with 2 children and 0.67 gluten factor 
; constant factor of 1.5
; random seed: 793030

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child119 - child
    bread391 bread22 - bread-portion
    content112 content181 - content-portion
    tray77 - tray
    table332 table474 table359 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread22)
     (at_kitchen_content content112)
     (at_kitchen_content content181)
     (no_gluten_bread bread391)
     (no_gluten_content content112)
     (allergic_gluten child119)
     (not_allergic_gluten child413)
     (waiting child413 table359)
     (waiting child119 table474)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child413)
     (served child119)
    )
  )
)
