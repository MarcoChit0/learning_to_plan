; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 982306

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 - child
    bread193 - bread-portion
    content96 - content-portion
    tray391 tray72 tray413 - tray
    table463 table320 table146 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray391 kitchen)
     (at tray72 kitchen)
     (at tray413 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_content content96)
     (not_allergic_gluten child128)
     (waiting child128 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child128)
    )
  )
)
