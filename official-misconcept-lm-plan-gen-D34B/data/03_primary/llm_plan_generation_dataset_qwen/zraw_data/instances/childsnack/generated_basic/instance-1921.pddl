; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 804977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 - child
    bread215 - bread-portion
    content48 - content-portion
    tray70 tray216 tray175 - tray
    table89 table237 table159 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray216 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread215)
     (at_kitchen_content content48)
     (not_allergic_gluten child87)
     (waiting child87 table159)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child87)
    )
  )
)
