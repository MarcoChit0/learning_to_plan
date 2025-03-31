; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 90603

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 - child
    bread438 - bread-portion
    content453 - content-portion
    tray499 - tray
    table393 table316 table64 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_content content453)
     (not_allergic_gluten child378)
     (waiting child378 table64)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child378)
    )
  )
)
