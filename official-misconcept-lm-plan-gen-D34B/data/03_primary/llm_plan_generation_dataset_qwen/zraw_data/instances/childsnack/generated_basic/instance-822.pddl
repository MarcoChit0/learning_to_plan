; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 576401

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 - child
    bread380 - bread-portion
    content121 - content-portion
    tray412 tray355 - tray
    table202 table390 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray412 kitchen)
     (at tray355 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_content content121)
     (not_allergic_gluten child274)
     (waiting child274 table390)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child274)
    )
  )
)
