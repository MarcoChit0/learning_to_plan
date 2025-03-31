; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 449578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child348 child83 - child
    bread461 bread200 bread149 - bread-portion
    content442 content359 content483 - content-portion
    tray360 tray431 tray208 tray209 - tray
    table287 table427 table71 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at tray431 kitchen)
     (at tray208 kitchen)
     (at tray209 kitchen)
     (at_kitchen_bread bread461)
     (at_kitchen_bread bread200)
     (at_kitchen_bread bread149)
     (at_kitchen_content content442)
     (at_kitchen_content content359)
     (at_kitchen_content content483)
     (not_allergic_gluten child83)
     (not_allergic_gluten child265)
     (not_allergic_gluten child348)
     (waiting child265 table427)
     (waiting child348 table71)
     (waiting child83 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child265)
     (served child348)
     (served child83)
    )
  )
)
