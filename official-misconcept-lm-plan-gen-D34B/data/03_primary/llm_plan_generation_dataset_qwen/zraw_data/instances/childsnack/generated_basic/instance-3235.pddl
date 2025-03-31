; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 22378

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child410 child356 - child
    bread244 bread423 - bread-portion
    content334 content9 - content-portion
    tray192 tray23 tray15 tray41 - tray
    table152 table359 table314 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at tray23 kitchen)
     (at tray15 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread244)
     (at_kitchen_bread bread423)
     (at_kitchen_content content334)
     (at_kitchen_content content9)
     (no_gluten_bread bread244)
     (no_gluten_content content334)
     (allergic_gluten child410)
     (not_allergic_gluten child356)
     (waiting child410 table314)
     (waiting child356 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child410)
     (served child356)
    )
  )
)
