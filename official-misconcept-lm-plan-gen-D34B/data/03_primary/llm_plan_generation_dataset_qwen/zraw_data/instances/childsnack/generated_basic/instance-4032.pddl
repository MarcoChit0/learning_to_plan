; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 162583

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 child231 child319 - child
    bread420 bread481 bread103 - bread-portion
    content458 content246 content250 - content-portion
    tray419 - tray
    table132 table380 table398 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray419 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread103)
     (at_kitchen_content content458)
     (at_kitchen_content content246)
     (at_kitchen_content content250)
     (not_allergic_gluten child319)
     (not_allergic_gluten child353)
     (not_allergic_gluten child231)
     (waiting child353 table398)
     (waiting child231 table398)
     (waiting child319 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child353)
     (served child231)
     (served child319)
    )
  )
)
