; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 567285

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 - child
    bread62 - bread-portion
    content368 - content-portion
    tray26 tray84 tray378 - tray
    table222 table48 table190 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray26 kitchen)
     (at tray84 kitchen)
     (at tray378 kitchen)
     (at_kitchen_bread bread62)
     (at_kitchen_content content368)
     (not_allergic_gluten child114)
     (waiting child114 table48)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child114)
    )
  )
)
