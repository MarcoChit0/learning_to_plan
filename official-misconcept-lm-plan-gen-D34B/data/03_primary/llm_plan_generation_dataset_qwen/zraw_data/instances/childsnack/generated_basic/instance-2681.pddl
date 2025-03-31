; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 838318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 - child
    bread132 - bread-portion
    content409 - content-portion
    tray466 tray277 tray3 tray259 - tray
    table322 table181 table431 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray277 kitchen)
     (at tray3 kitchen)
     (at tray259 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content409)
     (not_allergic_gluten child405)
     (waiting child405 table181)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child405)
    )
  )
)
