; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 190842

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 - child
    bread201 - bread-portion
    content57 - content-portion
    tray155 tray122 tray417 tray98 - tray
    table397 table222 table112 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at tray122 kitchen)
     (at tray417 kitchen)
     (at tray98 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_content content57)
     (not_allergic_gluten child495)
     (waiting child495 table397)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child495)
    )
  )
)
