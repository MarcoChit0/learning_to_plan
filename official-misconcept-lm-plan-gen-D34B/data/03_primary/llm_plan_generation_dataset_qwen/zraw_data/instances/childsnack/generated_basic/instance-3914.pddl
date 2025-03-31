; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 557529

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 child229 - child
    bread248 bread40 - bread-portion
    content243 content262 - content-portion
    tray130 - tray
    table323 table266 table123 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray130 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_bread bread40)
     (at_kitchen_content content243)
     (at_kitchen_content content262)
     (no_gluten_bread bread248)
     (no_gluten_content content262)
     (allergic_gluten child458)
     (not_allergic_gluten child229)
     (waiting child458 table323)
     (waiting child229 table323)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child458)
     (served child229)
    )
  )
)
