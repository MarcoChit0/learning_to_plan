; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 616788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 - child
    bread401 - bread-portion
    content123 - content-portion
    tray423 tray63 tray44 - tray
    table75 table254 table442 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at tray63 kitchen)
     (at tray44 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_content content123)
     (not_allergic_gluten child32)
     (waiting child32 table75)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child32)
    )
  )
)
