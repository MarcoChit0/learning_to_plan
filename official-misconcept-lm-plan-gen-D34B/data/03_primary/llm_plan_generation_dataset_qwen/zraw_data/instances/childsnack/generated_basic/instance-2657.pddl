; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 891694

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 - child
    bread421 - bread-portion
    content144 - content-portion
    tray18 tray346 tray288 tray469 - tray
    table347 table324 table96 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray346 kitchen)
     (at tray288 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread421)
     (at_kitchen_content content144)
     (not_allergic_gluten child95)
     (waiting child95 table96)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child95)
    )
  )
)
