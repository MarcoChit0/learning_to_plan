; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 647586

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child178 - child
    bread419 - bread-portion
    content462 - content-portion
    tray173 tray272 - tray
    table398 table457 table121 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray272 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_content content462)
     (not_allergic_gluten child178)
     (waiting child178 table121)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child178)
    )
  )
)
