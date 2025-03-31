; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 888737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 child262 - child
    bread221 bread388 - bread-portion
    content380 content490 - content-portion
    tray61 tray18 tray109 - tray
    table223 table182 table290 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray18 kitchen)
     (at tray109 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread388)
     (at_kitchen_content content380)
     (at_kitchen_content content490)
     (no_gluten_bread bread221)
     (no_gluten_content content490)
     (allergic_gluten child262)
     (not_allergic_gluten child261)
     (waiting child261 table290)
     (waiting child262 table182)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child261)
     (served child262)
    )
  )
)
