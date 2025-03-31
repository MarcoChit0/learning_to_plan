; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 696462

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 child238 - child
    bread470 bread29 - bread-portion
    content194 content60 - content-portion
    tray283 tray74 tray332 - tray
    table238 table79 table462 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray74 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread470)
     (at_kitchen_bread bread29)
     (at_kitchen_content content194)
     (at_kitchen_content content60)
     (no_gluten_bread bread29)
     (no_gluten_content content194)
     (allergic_gluten child427)
     (not_allergic_gluten child238)
     (waiting child427 table462)
     (waiting child238 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child427)
     (served child238)
    )
  )
)
