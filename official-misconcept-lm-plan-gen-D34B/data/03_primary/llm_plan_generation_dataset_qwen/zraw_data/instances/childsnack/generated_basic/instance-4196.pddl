; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 525350

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child429 child117 - child
    bread192 bread107 - bread-portion
    content97 content99 - content-portion
    tray250 - tray
    table243 table59 table284 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread107)
     (at_kitchen_content content97)
     (at_kitchen_content content99)
     (not_allergic_gluten child117)
     (not_allergic_gluten child429)
     (waiting child429 table284)
     (waiting child117 table59)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child429)
     (served child117)
    )
  )
)
