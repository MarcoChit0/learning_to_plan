; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 707996

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child243 - child
    bread368 bread389 - bread-portion
    content317 content293 - content-portion
    tray83 tray333 tray152 - tray
    table86 table80 table341 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray333 kitchen)
     (at tray152 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread389)
     (at_kitchen_content content317)
     (at_kitchen_content content293)
     (no_gluten_bread bread389)
     (no_gluten_content content293)
     (allergic_gluten child274)
     (not_allergic_gluten child243)
     (waiting child274 table341)
     (waiting child243 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child274)
     (served child243)
    )
  )
)
