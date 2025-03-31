; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 9770

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child61 - child
    bread105 - bread-portion
    content17 - content-portion
    tray410 tray431 tray360 - tray
    table386 table303 table167 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray431 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_content content17)
     (not_allergic_gluten child61)
     (waiting child61 table386)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child61)
    )
  )
)
