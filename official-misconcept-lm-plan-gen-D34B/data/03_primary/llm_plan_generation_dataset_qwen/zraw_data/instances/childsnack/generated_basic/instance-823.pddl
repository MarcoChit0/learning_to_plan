; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 505499

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 - child
    bread378 - bread-portion
    content375 - content-portion
    tray235 tray315 - tray
    table116 table365 table323 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray315 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_content content375)
     (not_allergic_gluten child239)
     (waiting child239 table323)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child239)
    )
  )
)
