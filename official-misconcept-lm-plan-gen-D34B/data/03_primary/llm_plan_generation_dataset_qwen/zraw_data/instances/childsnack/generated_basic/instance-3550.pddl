; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 113491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child464 - child
    bread22 bread494 - bread-portion
    content181 content167 - content-portion
    tray405 tray344 - tray
    table20 table284 table114 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray405 kitchen)
     (at tray344 kitchen)
     (at_kitchen_bread bread22)
     (at_kitchen_bread bread494)
     (at_kitchen_content content181)
     (at_kitchen_content content167)
     (not_allergic_gluten child131)
     (not_allergic_gluten child464)
     (waiting child131 table284)
     (waiting child464 table20)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child131)
     (served child464)
    )
  )
)
