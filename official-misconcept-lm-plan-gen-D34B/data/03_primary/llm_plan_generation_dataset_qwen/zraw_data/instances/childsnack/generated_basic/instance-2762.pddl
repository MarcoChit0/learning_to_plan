; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 460864

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child75 child337 - child
    bread131 bread450 - bread-portion
    content85 content369 - content-portion
    tray292 - tray
    table54 table261 table238 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_bread bread450)
     (at_kitchen_content content85)
     (at_kitchen_content content369)
     (no_gluten_bread bread131)
     (no_gluten_content content85)
     (allergic_gluten child337)
     (not_allergic_gluten child75)
     (waiting child75 table238)
     (waiting child337 table238)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child75)
     (served child337)
    )
  )
)
