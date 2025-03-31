; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 741076

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child314 - child
    bread115 bread479 - bread-portion
    content185 content434 - content-portion
    tray199 tray128 tray126 tray413 - tray
    table229 table131 table166 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray128 kitchen)
     (at tray126 kitchen)
     (at tray413 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread479)
     (at_kitchen_content content185)
     (at_kitchen_content content434)
     (no_gluten_bread bread115)
     (no_gluten_content content434)
     (allergic_gluten child314)
     (not_allergic_gluten child84)
     (waiting child84 table166)
     (waiting child314 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child84)
     (served child314)
    )
  )
)
