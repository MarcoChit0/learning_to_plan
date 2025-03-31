; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 774042

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 - child
    bread331 - bread-portion
    content489 - content-portion
    tray209 tray320 tray487 - tray
    table431 table418 table249 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray320 kitchen)
     (at tray487 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content489)
     (not_allergic_gluten child270)
     (waiting child270 table431)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child270)
    )
  )
)
