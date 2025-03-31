; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 204014

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 - child
    bread102 - bread-portion
    content165 - content-portion
    tray175 tray35 tray228 - tray
    table480 table397 table123 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray35 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content165)
     (not_allergic_gluten child295)
     (waiting child295 table123)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child295)
    )
  )
)
