; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 339048

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child364 child227 - child
    bread61 bread29 - bread-portion
    content321 content135 - content-portion
    tray423 - tray
    table69 table398 table162 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_bread bread29)
     (at_kitchen_content content321)
     (at_kitchen_content content135)
     (not_allergic_gluten child364)
     (not_allergic_gluten child227)
     (waiting child364 table398)
     (waiting child227 table162)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child364)
     (served child227)
    )
  )
)
