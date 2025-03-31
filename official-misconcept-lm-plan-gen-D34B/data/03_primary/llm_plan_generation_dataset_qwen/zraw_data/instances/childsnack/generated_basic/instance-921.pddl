; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 508191

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child482 child394 child167 - child
    bread349 bread147 bread191 - bread-portion
    content428 content34 content271 - content-portion
    tray196 - tray
    table378 table189 table339 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread191)
     (at_kitchen_content content428)
     (at_kitchen_content content34)
     (at_kitchen_content content271)
     (not_allergic_gluten child482)
     (not_allergic_gluten child167)
     (not_allergic_gluten child394)
     (waiting child482 table339)
     (waiting child394 table378)
     (waiting child167 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child482)
     (served child394)
     (served child167)
    )
  )
)
