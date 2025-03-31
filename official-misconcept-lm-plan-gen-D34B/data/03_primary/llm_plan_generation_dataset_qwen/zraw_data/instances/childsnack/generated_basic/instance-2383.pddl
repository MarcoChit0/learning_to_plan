; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 795854

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 child75 - child
    bread36 bread379 - bread-portion
    content490 content232 - content-portion
    tray491 - tray
    table133 table389 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_bread bread379)
     (at_kitchen_content content490)
     (at_kitchen_content content232)
     (not_allergic_gluten child75)
     (not_allergic_gluten child97)
     (waiting child97 table133)
     (waiting child75 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child97)
     (served child75)
    )
  )
)
