; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 580922

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 child286 - child
    bread33 bread183 - bread-portion
    content310 content380 - content-portion
    tray8 tray373 tray111 - tray
    table173 table414 table475 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray373 kitchen)
     (at tray111 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread183)
     (at_kitchen_content content310)
     (at_kitchen_content content380)
     (not_allergic_gluten child137)
     (not_allergic_gluten child286)
     (waiting child137 table173)
     (waiting child286 table475)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child137)
     (served child286)
    )
  )
)
