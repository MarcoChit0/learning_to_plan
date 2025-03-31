; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 91171

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 child294 child192 - child
    bread238 bread29 bread227 - bread-portion
    content49 content94 content33 - content-portion
    tray485 - tray
    table160 table459 table81 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread29)
     (at_kitchen_bread bread227)
     (at_kitchen_content content49)
     (at_kitchen_content content94)
     (at_kitchen_content content33)
     (not_allergic_gluten child152)
     (not_allergic_gluten child294)
     (not_allergic_gluten child192)
     (waiting child152 table459)
     (waiting child294 table459)
     (waiting child192 table81)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child152)
     (served child294)
     (served child192)
    )
  )
)
