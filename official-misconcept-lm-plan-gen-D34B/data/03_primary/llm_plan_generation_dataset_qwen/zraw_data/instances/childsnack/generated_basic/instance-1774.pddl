; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 79480

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child337 - child
    bread120 - bread-portion
    content196 - content-portion
    tray459 - tray
    table479 table431 table86 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_content content196)
     (not_allergic_gluten child337)
     (waiting child337 table86)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child337)
    )
  )
)
