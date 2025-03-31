; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 534256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 - child
    bread485 - bread-portion
    content488 - content-portion
    tray372 tray118 tray486 tray80 - tray
    table378 table495 table472 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray372 kitchen)
     (at tray118 kitchen)
     (at tray486 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_content content488)
     (not_allergic_gluten child318)
     (waiting child318 table472)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child318)
    )
  )
)
