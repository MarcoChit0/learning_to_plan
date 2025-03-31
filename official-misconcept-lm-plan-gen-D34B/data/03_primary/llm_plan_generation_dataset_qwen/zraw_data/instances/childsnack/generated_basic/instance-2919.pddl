; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 303463

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 child266 - child
    bread258 bread167 - bread-portion
    content118 content8 - content-portion
    tray247 tray3 tray392 - tray
    table160 table244 table332 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray247 kitchen)
     (at tray3 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread167)
     (at_kitchen_content content118)
     (at_kitchen_content content8)
     (no_gluten_bread bread167)
     (no_gluten_content content118)
     (allergic_gluten child266)
     (not_allergic_gluten child441)
     (waiting child441 table244)
     (waiting child266 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child441)
     (served child266)
    )
  )
)
