; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 171267

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 - child
    bread467 - bread-portion
    content271 - content-portion
    tray459 tray207 tray67 - tray
    table103 table414 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray207 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_content content271)
     (not_allergic_gluten child315)
     (waiting child315 table136)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child315)
    )
  )
)
