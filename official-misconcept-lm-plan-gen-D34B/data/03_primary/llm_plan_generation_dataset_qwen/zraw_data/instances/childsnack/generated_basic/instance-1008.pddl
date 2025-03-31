; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 693623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child252 child71 - child
    bread60 bread285 bread31 - bread-portion
    content351 content283 content147 - content-portion
    tray403 - tray
    table494 table188 table171 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread31)
     (at_kitchen_content content351)
     (at_kitchen_content content283)
     (at_kitchen_content content147)
     (not_allergic_gluten child71)
     (not_allergic_gluten child294)
     (not_allergic_gluten child252)
     (waiting child294 table494)
     (waiting child252 table171)
     (waiting child71 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child294)
     (served child252)
     (served child71)
    )
  )
)
