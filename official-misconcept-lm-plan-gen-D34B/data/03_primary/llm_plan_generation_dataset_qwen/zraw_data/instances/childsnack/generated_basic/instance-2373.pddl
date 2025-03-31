; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 186461

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child434 - child
    bread284 bread457 - bread-portion
    content414 content452 - content-portion
    tray303 - tray
    table228 table135 table343 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread457)
     (at_kitchen_content content414)
     (at_kitchen_content content452)
     (not_allergic_gluten child16)
     (not_allergic_gluten child434)
     (waiting child16 table343)
     (waiting child434 table228)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child16)
     (served child434)
    )
  )
)
