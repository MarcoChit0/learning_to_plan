; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 676075

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child143 - child
    bread69 bread285 - bread-portion
    content258 content472 - content-portion
    tray208 - tray
    table21 table391 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray208 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_bread bread285)
     (at_kitchen_content content258)
     (at_kitchen_content content472)
     (no_gluten_bread bread69)
     (no_gluten_content content258)
     (allergic_gluten child120)
     (not_allergic_gluten child143)
     (waiting child120 table21)
     (waiting child143 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child143)
    )
  )
)
