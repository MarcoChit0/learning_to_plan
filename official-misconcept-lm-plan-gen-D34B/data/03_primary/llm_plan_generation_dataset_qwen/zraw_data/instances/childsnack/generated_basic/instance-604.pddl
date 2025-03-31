; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 887437

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 - child
    bread260 - bread-portion
    content396 - content-portion
    tray111 - tray
    table253 table109 table170 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray111 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_content content396)
     (not_allergic_gluten child63)
     (waiting child63 table170)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child63)
    )
  )
)
