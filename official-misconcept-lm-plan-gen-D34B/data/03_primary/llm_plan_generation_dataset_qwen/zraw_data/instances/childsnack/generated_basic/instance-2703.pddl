; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 951186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child44 child114 child228 - child
    bread24 bread321 bread135 - bread-portion
    content287 content418 content292 - content-portion
    tray357 tray449 tray23 - tray
    table237 table142 table150 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray449 kitchen)
     (at tray23 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread135)
     (at_kitchen_content content287)
     (at_kitchen_content content418)
     (at_kitchen_content content292)
     (not_allergic_gluten child44)
     (not_allergic_gluten child228)
     (not_allergic_gluten child114)
     (waiting child44 table142)
     (waiting child114 table237)
     (waiting child228 table150)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child44)
     (served child114)
     (served child228)
    )
  )
)
