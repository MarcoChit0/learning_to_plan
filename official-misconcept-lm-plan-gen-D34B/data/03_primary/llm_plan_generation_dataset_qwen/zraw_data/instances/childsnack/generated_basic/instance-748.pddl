; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 69003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child320 - child
    bread49 bread55 - bread-portion
    content384 content357 - content-portion
    tray432 tray292 - tray
    table385 table348 table451 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray292 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread55)
     (at_kitchen_content content384)
     (at_kitchen_content content357)
     (no_gluten_bread bread55)
     (no_gluten_content content357)
     (allergic_gluten child142)
     (not_allergic_gluten child320)
     (waiting child142 table451)
     (waiting child320 table385)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child142)
     (served child320)
    )
  )
)
