; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 272644

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child152 - child
    bread294 bread160 - bread-portion
    content296 content431 - content-portion
    tray35 tray416 - tray
    table273 table102 table37 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray416 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread160)
     (at_kitchen_content content296)
     (at_kitchen_content content431)
     (no_gluten_bread bread160)
     (no_gluten_content content431)
     (allergic_gluten child152)
     (not_allergic_gluten child478)
     (waiting child478 table102)
     (waiting child152 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child152)
    )
  )
)
