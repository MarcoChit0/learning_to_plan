; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 141121

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child74 child142 - child
    bread489 bread162 - bread-portion
    content131 content327 - content-portion
    tray333 tray94 - tray
    table423 table299 table206 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray94 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread162)
     (at_kitchen_content content131)
     (at_kitchen_content content327)
     (no_gluten_bread bread162)
     (no_gluten_content content327)
     (allergic_gluten child74)
     (not_allergic_gluten child142)
     (waiting child74 table423)
     (waiting child142 table423)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child74)
     (served child142)
    )
  )
)
