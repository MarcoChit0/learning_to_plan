; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 453310

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child230 child459 - child
    bread130 bread446 - bread-portion
    content80 content469 - content-portion
    tray157 - tray
    table304 table498 table267 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at_kitchen_bread bread130)
     (at_kitchen_bread bread446)
     (at_kitchen_content content80)
     (at_kitchen_content content469)
     (no_gluten_bread bread446)
     (no_gluten_content content80)
     (allergic_gluten child230)
     (not_allergic_gluten child459)
     (waiting child230 table498)
     (waiting child459 table267)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child230)
     (served child459)
    )
  )
)
