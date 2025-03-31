; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 34668

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child242 - child
    bread356 bread358 - bread-portion
    content374 content360 - content-portion
    tray305 - tray
    table312 table182 table397 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_bread bread358)
     (at_kitchen_content content374)
     (at_kitchen_content content360)
     (no_gluten_bread bread356)
     (no_gluten_content content360)
     (allergic_gluten child279)
     (not_allergic_gluten child242)
     (waiting child279 table182)
     (waiting child242 table182)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child279)
     (served child242)
    )
  )
)
