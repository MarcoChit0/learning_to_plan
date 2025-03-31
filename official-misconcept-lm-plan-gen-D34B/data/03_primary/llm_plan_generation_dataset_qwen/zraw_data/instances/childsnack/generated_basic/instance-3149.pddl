; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 546750

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 child58 - child
    bread72 bread145 - bread-portion
    content355 content408 - content-portion
    tray418 tray118 - tray
    table93 table297 table209 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray118 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread145)
     (at_kitchen_content content355)
     (at_kitchen_content content408)
     (not_allergic_gluten child58)
     (not_allergic_gluten child475)
     (waiting child475 table209)
     (waiting child58 table93)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child475)
     (served child58)
    )
  )
)
