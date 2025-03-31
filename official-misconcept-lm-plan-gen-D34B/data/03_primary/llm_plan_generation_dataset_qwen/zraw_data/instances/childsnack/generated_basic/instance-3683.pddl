; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 860871

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child340 child178 - child
    bread184 bread295 - bread-portion
    content273 content438 - content-portion
    tray252 tray288 - tray
    table440 table123 table50 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread295)
     (at_kitchen_content content273)
     (at_kitchen_content content438)
     (no_gluten_bread bread184)
     (no_gluten_content content438)
     (allergic_gluten child178)
     (not_allergic_gluten child340)
     (waiting child340 table50)
     (waiting child178 table123)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child340)
     (served child178)
    )
  )
)
