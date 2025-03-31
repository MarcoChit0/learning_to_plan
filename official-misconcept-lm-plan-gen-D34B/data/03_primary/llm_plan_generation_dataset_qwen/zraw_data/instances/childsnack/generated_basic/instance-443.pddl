; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 516541

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child454 child352 child54 - child
    bread259 bread152 bread98 - bread-portion
    content411 content435 content35 - content-portion
    tray157 - tray
    table240 table248 table327 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread98)
     (at_kitchen_content content411)
     (at_kitchen_content content435)
     (at_kitchen_content content35)
     (not_allergic_gluten child54)
     (not_allergic_gluten child352)
     (not_allergic_gluten child454)
     (waiting child454 table327)
     (waiting child352 table327)
     (waiting child54 table240)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child454)
     (served child352)
     (served child54)
    )
  )
)
