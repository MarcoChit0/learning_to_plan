; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 156674

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child61 - child
    bread302 - bread-portion
    content74 - content-portion
    tray279 tray317 - tray
    table62 table92 table31 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_content content74)
     (not_allergic_gluten child61)
     (waiting child61 table31)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child61)
    )
  )
)
