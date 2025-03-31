; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 814226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child235 child103 - child
    bread210 bread104 - bread-portion
    content247 content44 - content-portion
    tray481 tray5 - tray
    table270 table118 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray5 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_bread bread104)
     (at_kitchen_content content247)
     (at_kitchen_content content44)
     (not_allergic_gluten child103)
     (not_allergic_gluten child235)
     (waiting child235 table380)
     (waiting child103 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child235)
     (served child103)
    )
  )
)
