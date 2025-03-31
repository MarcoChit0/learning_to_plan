; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 571808

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child267 - child
    bread269 bread197 - bread-portion
    content45 content149 - content-portion
    tray21 - tray
    table68 table259 table12 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at_kitchen_bread bread269)
     (at_kitchen_bread bread197)
     (at_kitchen_content content45)
     (at_kitchen_content content149)
     (no_gluten_bread bread197)
     (no_gluten_content content45)
     (allergic_gluten child85)
     (not_allergic_gluten child267)
     (waiting child85 table259)
     (waiting child267 table68)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child267)
    )
  )
)
