; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 246704

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 - child
    bread260 - bread-portion
    content224 - content-portion
    tray358 tray27 tray458 tray76 - tray
    table125 table444 table489 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray27 kitchen)
     (at tray458 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_content content224)
     (not_allergic_gluten child308)
     (waiting child308 table125)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child308)
    )
  )
)
