; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 678174

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child0 child291 - child
    bread17 bread423 bread180 - bread-portion
    content350 content177 content370 - content-portion
    tray417 - tray
    table165 table88 table357 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray417 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_bread bread423)
     (at_kitchen_bread bread180)
     (at_kitchen_content content350)
     (at_kitchen_content content177)
     (at_kitchen_content content370)
     (not_allergic_gluten child0)
     (not_allergic_gluten child157)
     (not_allergic_gluten child291)
     (waiting child157 table357)
     (waiting child0 table357)
     (waiting child291 table88)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child157)
     (served child0)
     (served child291)
    )
  )
)
