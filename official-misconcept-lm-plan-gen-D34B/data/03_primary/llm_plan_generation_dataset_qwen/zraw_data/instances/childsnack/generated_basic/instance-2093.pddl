; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 692409

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 - child
    bread290 - bread-portion
    content394 - content-portion
    tray70 tray401 tray487 tray373 - tray
    table125 table311 table100 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray401 kitchen)
     (at tray487 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread290)
     (at_kitchen_content content394)
     (not_allergic_gluten child121)
     (waiting child121 table100)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child121)
    )
  )
)
