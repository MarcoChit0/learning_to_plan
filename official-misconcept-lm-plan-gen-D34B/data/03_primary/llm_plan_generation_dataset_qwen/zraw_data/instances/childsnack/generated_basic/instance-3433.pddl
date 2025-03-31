; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 291013

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 - child
    bread493 - bread-portion
    content169 - content-portion
    tray258 tray445 tray92 tray6 - tray
    table136 table368 table251 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray258 kitchen)
     (at tray445 kitchen)
     (at tray92 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread493)
     (at_kitchen_content content169)
     (not_allergic_gluten child157)
     (waiting child157 table251)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child157)
    )
  )
)
