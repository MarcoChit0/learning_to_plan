; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 933561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child106 - child
    bread410 bread479 - bread-portion
    content303 content490 - content-portion
    tray499 tray51 tray273 tray77 - tray
    table19 table417 table317 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray51 kitchen)
     (at tray273 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_bread bread479)
     (at_kitchen_content content303)
     (at_kitchen_content content490)
     (no_gluten_bread bread410)
     (no_gluten_content content490)
     (allergic_gluten child478)
     (not_allergic_gluten child106)
     (waiting child478 table19)
     (waiting child106 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child106)
    )
  )
)
