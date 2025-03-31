; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 61724

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 child36 - child
    bread369 bread161 - bread-portion
    content451 content231 - content-portion
    tray109 - tray
    table209 table322 table11 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread161)
     (at_kitchen_content content451)
     (at_kitchen_content content231)
     (no_gluten_bread bread161)
     (no_gluten_content content231)
     (allergic_gluten child291)
     (not_allergic_gluten child36)
     (waiting child291 table11)
     (waiting child36 table322)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child291)
     (served child36)
    )
  )
)
