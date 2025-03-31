; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 974515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child111 - child
    bread448 bread117 - bread-portion
    content477 content227 - content-portion
    tray98 - tray
    table437 table367 table144 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray98 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread117)
     (at_kitchen_content content477)
     (at_kitchen_content content227)
     (not_allergic_gluten child111)
     (not_allergic_gluten child431)
     (waiting child431 table437)
     (waiting child111 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child111)
    )
  )
)
