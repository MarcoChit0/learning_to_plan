; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 99886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 child230 - child
    bread426 bread11 - bread-portion
    content384 content68 - content-portion
    tray24 tray21 tray298 - tray
    table196 table302 table277 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray21 kitchen)
     (at tray298 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread11)
     (at_kitchen_content content384)
     (at_kitchen_content content68)
     (no_gluten_bread bread11)
     (no_gluten_content content68)
     (allergic_gluten child312)
     (not_allergic_gluten child230)
     (waiting child312 table196)
     (waiting child230 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child312)
     (served child230)
    )
  )
)
