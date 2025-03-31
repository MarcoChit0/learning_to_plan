; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 444964

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child348 - child
    bread276 bread343 - bread-portion
    content225 content281 - content-portion
    tray307 tray111 - tray
    table58 table273 table162 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray111 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread343)
     (at_kitchen_content content225)
     (at_kitchen_content content281)
     (no_gluten_bread bread276)
     (no_gluten_content content281)
     (allergic_gluten child348)
     (not_allergic_gluten child413)
     (waiting child413 table273)
     (waiting child348 table162)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child413)
     (served child348)
    )
  )
)
