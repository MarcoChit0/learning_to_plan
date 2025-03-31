; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 696996

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 - child
    bread86 - bread-portion
    content325 - content-portion
    tray460 tray425 tray442 - tray
    table88 table59 table98 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray460 kitchen)
     (at tray425 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content325)
     (not_allergic_gluten child252)
     (waiting child252 table59)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child252)
    )
  )
)
