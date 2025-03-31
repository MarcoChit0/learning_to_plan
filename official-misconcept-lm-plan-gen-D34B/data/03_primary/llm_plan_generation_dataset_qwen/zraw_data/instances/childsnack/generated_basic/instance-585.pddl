; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 483451

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 - child
    bread140 - bread-portion
    content149 - content-portion
    tray242 - tray
    table292 table361 table329 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at_kitchen_bread bread140)
     (at_kitchen_content content149)
     (not_allergic_gluten child50)
     (waiting child50 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child50)
    )
  )
)
