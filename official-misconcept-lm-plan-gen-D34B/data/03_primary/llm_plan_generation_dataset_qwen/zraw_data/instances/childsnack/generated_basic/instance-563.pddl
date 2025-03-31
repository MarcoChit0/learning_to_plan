; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 714646

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child209 - child
    bread45 bread146 - bread-portion
    content438 content282 - content-portion
    tray235 - tray
    table62 table96 table153 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread146)
     (at_kitchen_content content438)
     (at_kitchen_content content282)
     (no_gluten_bread bread146)
     (no_gluten_content content282)
     (allergic_gluten child209)
     (not_allergic_gluten child478)
     (waiting child478 table96)
     (waiting child209 table96)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child209)
    )
  )
)
