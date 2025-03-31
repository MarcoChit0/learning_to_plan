; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 465311

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child119 child82 - child
    bread6 bread246 - bread-portion
    content233 content187 - content-portion
    tray312 tray213 - tray
    table122 table19 table101 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray312 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread6)
     (at_kitchen_bread bread246)
     (at_kitchen_content content233)
     (at_kitchen_content content187)
     (no_gluten_bread bread6)
     (no_gluten_content content187)
     (allergic_gluten child82)
     (not_allergic_gluten child119)
     (waiting child119 table122)
     (waiting child82 table19)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child119)
     (served child82)
    )
  )
)
