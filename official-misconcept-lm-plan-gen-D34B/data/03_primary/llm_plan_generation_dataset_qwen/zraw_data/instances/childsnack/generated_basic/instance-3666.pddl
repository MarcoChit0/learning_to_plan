; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 879209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child83 - child
    bread282 bread242 - bread-portion
    content432 content193 - content-portion
    tray271 tray420 - tray
    table159 table199 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread282)
     (at_kitchen_bread bread242)
     (at_kitchen_content content432)
     (at_kitchen_content content193)
     (no_gluten_bread bread282)
     (no_gluten_content content193)
     (allergic_gluten child83)
     (not_allergic_gluten child175)
     (waiting child175 table159)
     (waiting child83 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child175)
     (served child83)
    )
  )
)
