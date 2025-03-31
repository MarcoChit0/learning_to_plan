; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 172504

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 - child
    bread102 - bread-portion
    content50 - content-portion
    tray488 tray76 - tray
    table58 table228 table41 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content50)
     (not_allergic_gluten child450)
     (waiting child450 table41)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child450)
    )
  )
)
