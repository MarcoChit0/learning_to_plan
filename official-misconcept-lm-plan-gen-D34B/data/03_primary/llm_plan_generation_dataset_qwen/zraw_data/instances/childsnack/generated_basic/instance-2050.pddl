; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 378249

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child374 child317 - child
    bread311 bread217 - bread-portion
    content384 content379 - content-portion
    tray157 tray490 tray360 - tray
    table223 table128 table364 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray490 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_bread bread217)
     (at_kitchen_content content384)
     (at_kitchen_content content379)
     (not_allergic_gluten child374)
     (not_allergic_gluten child317)
     (waiting child374 table128)
     (waiting child317 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child374)
     (served child317)
    )
  )
)
