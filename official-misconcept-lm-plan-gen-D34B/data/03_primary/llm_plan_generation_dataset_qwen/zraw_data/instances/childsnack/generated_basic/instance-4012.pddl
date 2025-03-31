; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 483453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child439 child353 - child
    bread12 bread243 bread17 - bread-portion
    content408 content13 content370 - content-portion
    tray167 - tray
    table4 table382 table170 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread17)
     (at_kitchen_content content408)
     (at_kitchen_content content13)
     (at_kitchen_content content370)
     (not_allergic_gluten child439)
     (not_allergic_gluten child353)
     (not_allergic_gluten child436)
     (waiting child436 table4)
     (waiting child439 table382)
     (waiting child353 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child436)
     (served child439)
     (served child353)
    )
  )
)
