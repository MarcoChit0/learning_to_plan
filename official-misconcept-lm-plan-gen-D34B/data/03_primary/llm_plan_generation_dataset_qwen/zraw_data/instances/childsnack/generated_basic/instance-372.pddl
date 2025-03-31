; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 379861

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread433 - bread-portion
    content195 - content-portion
    tray401 tray437 tray493 tray229 - tray
    table342 table253 table313 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at tray437 kitchen)
     (at tray493 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_content content195)
     (not_allergic_gluten child156)
     (waiting child156 table253)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
