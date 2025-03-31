; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 519435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 - child
    bread400 - bread-portion
    content485 - content-portion
    tray256 - tray
    table283 table139 table52 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content485)
     (not_allergic_gluten child433)
     (waiting child433 table139)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child433)
    )
  )
)
