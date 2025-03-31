; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 681043

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 child253 - child
    bread249 bread499 - bread-portion
    content226 content314 - content-portion
    tray170 - tray
    table184 table427 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at_kitchen_bread bread249)
     (at_kitchen_bread bread499)
     (at_kitchen_content content226)
     (at_kitchen_content content314)
     (not_allergic_gluten child253)
     (not_allergic_gluten child267)
     (waiting child267 table478)
     (waiting child253 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child267)
     (served child253)
    )
  )
)
