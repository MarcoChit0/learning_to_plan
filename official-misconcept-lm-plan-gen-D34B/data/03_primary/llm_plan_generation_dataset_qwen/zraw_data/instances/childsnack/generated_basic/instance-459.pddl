; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 611693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child101 child26 child73 - child
    bread445 bread489 bread457 - bread-portion
    content467 content9 content228 - content-portion
    tray475 - tray
    table148 table332 table158 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray475 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread457)
     (at_kitchen_content content467)
     (at_kitchen_content content9)
     (at_kitchen_content content228)
     (not_allergic_gluten child26)
     (not_allergic_gluten child101)
     (not_allergic_gluten child73)
     (waiting child101 table332)
     (waiting child26 table332)
     (waiting child73 table332)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child101)
     (served child26)
     (served child73)
    )
  )
)
