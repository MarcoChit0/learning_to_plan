; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 271545

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 - child
    bread410 - bread-portion
    content205 - content-portion
    tray31 tray186 - tray
    table321 table273 table439 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray31 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_content content205)
     (not_allergic_gluten child185)
     (waiting child185 table321)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child185)
    )
  )
)
