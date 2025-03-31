; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 600682

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child357 child174 - child
    bread211 bread45 bread309 - bread-portion
    content203 content272 content55 - content-portion
    tray80 tray59 tray75 - tray
    table349 table171 table435 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray59 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread309)
     (at_kitchen_content content203)
     (at_kitchen_content content272)
     (at_kitchen_content content55)
     (not_allergic_gluten child357)
     (not_allergic_gluten child167)
     (not_allergic_gluten child174)
     (waiting child167 table435)
     (waiting child357 table171)
     (waiting child174 table435)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child167)
     (served child357)
     (served child174)
    )
  )
)
