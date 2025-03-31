; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 682941

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child268 - child
    bread18 - bread-portion
    content453 - content-portion
    tray121 tray133 tray118 - tray
    table297 table398 table279 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray121 kitchen)
     (at tray133 kitchen)
     (at tray118 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_content content453)
     (not_allergic_gluten child268)
     (waiting child268 table398)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child268)
    )
  )
)
