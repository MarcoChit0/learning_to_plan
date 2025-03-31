; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 953602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 - child
    bread138 - bread-portion
    content119 - content-portion
    tray241 tray52 tray261 - tray
    table436 table22 table293 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray52 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_content content119)
     (not_allergic_gluten child267)
     (waiting child267 table22)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child267)
    )
  )
)
