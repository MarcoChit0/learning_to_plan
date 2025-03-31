; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 125453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child139 child166 - child
    bread118 bread222 - bread-portion
    content304 content25 - content-portion
    tray41 tray161 - tray
    table34 table269 table294 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray41 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread118)
     (at_kitchen_bread bread222)
     (at_kitchen_content content304)
     (at_kitchen_content content25)
     (not_allergic_gluten child139)
     (not_allergic_gluten child166)
     (waiting child139 table294)
     (waiting child166 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child139)
     (served child166)
    )
  )
)
