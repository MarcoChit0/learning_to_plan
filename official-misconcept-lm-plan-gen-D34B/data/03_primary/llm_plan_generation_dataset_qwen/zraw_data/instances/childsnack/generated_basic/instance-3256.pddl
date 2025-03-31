; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 282897

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child299 - child
    bread308 bread256 - bread-portion
    content90 content50 - content-portion
    tray55 tray400 tray405 tray95 - tray
    table488 table80 table398 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray400 kitchen)
     (at tray405 kitchen)
     (at tray95 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread256)
     (at_kitchen_content content90)
     (at_kitchen_content content50)
     (no_gluten_bread bread256)
     (no_gluten_content content50)
     (allergic_gluten child237)
     (not_allergic_gluten child299)
     (waiting child237 table80)
     (waiting child299 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child237)
     (served child299)
    )
  )
)
