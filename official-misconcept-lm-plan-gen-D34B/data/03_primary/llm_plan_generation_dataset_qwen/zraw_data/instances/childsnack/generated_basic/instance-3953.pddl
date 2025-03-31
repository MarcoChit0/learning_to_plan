; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 777475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child258 - child
    bread147 bread211 - bread-portion
    content307 content62 - content-portion
    tray52 - tray
    table347 table204 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread211)
     (at_kitchen_content content307)
     (at_kitchen_content content62)
     (no_gluten_bread bread147)
     (no_gluten_content content62)
     (allergic_gluten child244)
     (not_allergic_gluten child258)
     (waiting child244 table243)
     (waiting child258 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child244)
     (served child258)
    )
  )
)
