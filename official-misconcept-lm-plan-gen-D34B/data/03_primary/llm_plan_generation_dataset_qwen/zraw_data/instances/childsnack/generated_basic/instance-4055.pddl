; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 738663

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 child212 child64 - child
    bread296 bread182 bread91 - bread-portion
    content167 content276 content458 - content-portion
    tray243 - tray
    table496 table390 table395 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread182)
     (at_kitchen_bread bread91)
     (at_kitchen_content content167)
     (at_kitchen_content content276)
     (at_kitchen_content content458)
     (not_allergic_gluten child486)
     (not_allergic_gluten child64)
     (not_allergic_gluten child212)
     (waiting child486 table395)
     (waiting child212 table390)
     (waiting child64 table496)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child486)
     (served child212)
     (served child64)
    )
  )
)
