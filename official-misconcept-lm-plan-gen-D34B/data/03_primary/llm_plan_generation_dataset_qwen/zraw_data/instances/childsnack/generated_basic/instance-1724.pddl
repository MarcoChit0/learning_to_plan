; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 334890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 - child
    bread46 - bread-portion
    content119 - content-portion
    tray461 tray384 tray41 - tray
    table151 table292 table142 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray461 kitchen)
     (at tray384 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_content content119)
     (not_allergic_gluten child4)
     (waiting child4 table142)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child4)
    )
  )
)
