; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 602296

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 - child
    bread498 - bread-portion
    content253 - content-portion
    tray15 - tray
    table39 table124 table92 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content253)
     (not_allergic_gluten child266)
     (waiting child266 table39)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child266)
    )
  )
)
