; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 253767

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 - child
    bread164 - bread-portion
    content312 - content-portion
    tray349 tray412 - tray
    table80 table57 table256 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_content content312)
     (not_allergic_gluten child484)
     (waiting child484 table57)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child484)
    )
  )
)
