; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 704964

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child213 child78 - child
    bread94 bread300 - bread-portion
    content89 content457 - content-portion
    tray181 tray303 tray217 - tray
    table88 table245 table244 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray303 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread300)
     (at_kitchen_content content89)
     (at_kitchen_content content457)
     (no_gluten_bread bread94)
     (no_gluten_content content457)
     (allergic_gluten child213)
     (not_allergic_gluten child78)
     (waiting child213 table88)
     (waiting child78 table245)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child213)
     (served child78)
    )
  )
)
