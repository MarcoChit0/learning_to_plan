; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 383533

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 - child
    bread144 - bread-portion
    content273 - content-portion
    tray495 tray497 tray229 - tray
    table220 table197 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray497 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_content content273)
     (not_allergic_gluten child210)
     (waiting child210 table136)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child210)
    )
  )
)
