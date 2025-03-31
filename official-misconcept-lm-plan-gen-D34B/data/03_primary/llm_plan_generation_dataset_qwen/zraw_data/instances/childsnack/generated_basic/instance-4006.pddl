; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 567219

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 - child
    bread98 - bread-portion
    content383 - content-portion
    tray294 tray183 tray328 - tray
    table242 table116 table232 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at tray183 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread98)
     (at_kitchen_content content383)
     (not_allergic_gluten child66)
     (waiting child66 table232)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child66)
    )
  )
)
