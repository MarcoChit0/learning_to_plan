; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 747248

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child429 - child
    bread457 - bread-portion
    content102 - content-portion
    tray257 tray402 - tray
    table430 table52 table304 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray257 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_content content102)
     (not_allergic_gluten child429)
     (waiting child429 table430)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child429)
    )
  )
)
