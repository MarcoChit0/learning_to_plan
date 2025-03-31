; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 921096

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child392 child163 child70 - child
    bread30 bread171 bread262 - bread-portion
    content399 content405 content177 - content-portion
    tray179 tray374 tray144 - tray
    table491 table300 table486 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray179 kitchen)
     (at tray374 kitchen)
     (at tray144 kitchen)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread171)
     (at_kitchen_bread bread262)
     (at_kitchen_content content399)
     (at_kitchen_content content405)
     (at_kitchen_content content177)
     (not_allergic_gluten child70)
     (not_allergic_gluten child163)
     (not_allergic_gluten child392)
     (waiting child392 table300)
     (waiting child163 table486)
     (waiting child70 table486)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child392)
     (served child163)
     (served child70)
    )
  )
)
