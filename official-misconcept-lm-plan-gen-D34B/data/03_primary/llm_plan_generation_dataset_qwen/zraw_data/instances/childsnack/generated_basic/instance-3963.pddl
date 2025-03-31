; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 513852

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 - child
    bread204 - bread-portion
    content229 - content-portion
    tray22 tray114 tray495 - tray
    table171 table220 table289 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at tray114 kitchen)
     (at tray495 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_content content229)
     (not_allergic_gluten child148)
     (waiting child148 table171)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child148)
    )
  )
)
