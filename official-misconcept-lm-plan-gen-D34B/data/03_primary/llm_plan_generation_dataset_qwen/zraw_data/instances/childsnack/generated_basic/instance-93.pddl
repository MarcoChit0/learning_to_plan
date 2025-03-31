; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 515004

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 child182 - child
    bread98 bread203 - bread-portion
    content177 content310 - content-portion
    tray198 tray348 - tray
    table338 table462 table387 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray348 kitchen)
     (at_kitchen_bread bread98)
     (at_kitchen_bread bread203)
     (at_kitchen_content content177)
     (at_kitchen_content content310)
     (not_allergic_gluten child438)
     (not_allergic_gluten child182)
     (waiting child438 table338)
     (waiting child182 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child438)
     (served child182)
    )
  )
)
