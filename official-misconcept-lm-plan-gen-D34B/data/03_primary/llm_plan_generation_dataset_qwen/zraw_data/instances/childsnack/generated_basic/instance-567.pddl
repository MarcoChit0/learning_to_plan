; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 539521

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 child162 - child
    bread476 bread229 - bread-portion
    content189 content251 - content-portion
    tray254 - tray
    table189 table347 table329 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray254 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread229)
     (at_kitchen_content content189)
     (at_kitchen_content content251)
     (no_gluten_bread bread229)
     (no_gluten_content content189)
     (allergic_gluten child95)
     (not_allergic_gluten child162)
     (waiting child95 table347)
     (waiting child162 table189)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child95)
     (served child162)
    )
  )
)
