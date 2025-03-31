; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 356329

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 - child
    bread339 - bread-portion
    content26 - content-portion
    tray457 tray340 tray415 tray231 - tray
    table401 table70 table496 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at tray340 kitchen)
     (at tray415 kitchen)
     (at tray231 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_content content26)
     (not_allergic_gluten child55)
     (waiting child55 table401)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child55)
    )
  )
)
