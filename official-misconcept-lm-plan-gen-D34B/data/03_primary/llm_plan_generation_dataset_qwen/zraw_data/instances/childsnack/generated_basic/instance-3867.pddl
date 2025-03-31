; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 625752

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 - child
    bread57 - bread-portion
    content415 - content-portion
    tray376 tray412 - tray
    table410 table467 table385 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_content content415)
     (not_allergic_gluten child349)
     (waiting child349 table410)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child349)
    )
  )
)
