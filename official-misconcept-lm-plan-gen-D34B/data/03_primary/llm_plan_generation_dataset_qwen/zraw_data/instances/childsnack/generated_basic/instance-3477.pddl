; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 600593

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child149 - child
    bread451 bread313 - bread-portion
    content393 content165 - content-portion
    tray36 tray365 - tray
    table320 table30 table333 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread313)
     (at_kitchen_content content393)
     (at_kitchen_content content165)
     (no_gluten_bread bread451)
     (no_gluten_content content393)
     (allergic_gluten child202)
     (not_allergic_gluten child149)
     (waiting child202 table333)
     (waiting child149 table320)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child202)
     (served child149)
    )
  )
)
